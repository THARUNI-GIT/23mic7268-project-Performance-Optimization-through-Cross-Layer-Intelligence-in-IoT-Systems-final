/*
 * ACLIF NS3 Simulation Script
 * Result 10: DQN Convergence
 * 
 * Framework : Autonomous Cross-Layer Intelligence Framework (ACLIF)
 * Author    : B. Tharuni Sri Sai, VIT-AP University
 * ORCID     : 0009-0004-9561-8985
 * Simulator : NS3 v3.38
 *
 * Compile:  ./ns3 build
 * Run:      ./ns3 run "scratch/dqn_convergence --episode=100"
 *
 * Sweeps: episode over [1..300]
 * Output metric: reward_epsilon
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/energy-module.h"
#include "ns3/lr-wpan-module.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("ACLIF_Res10_dqn_convergence");

// ─── ACLIF Global Parameters ─────────────────────────────────────────────────
static const uint32_t DEFAULT_N_NODES   = 100;
static const double   AREA_SIDE         = 200.0;   // metres
static const double   SIM_TIME          = 300.0;   // seconds
static const double   CTRL_INTERVAL     = 0.5;     // seconds (500 ms)
static const uint32_t PKT_SIZE          = 512;     // bytes
static const double   PKT_RATE_DEFAULT  = 5.0;     // pkt/s
static const double   INIT_ENERGY       = 2.0;     // Joules
static const double   E_ELEC            = 50e-9;   // J/bit
static const double   E_AMP             = 1.3e-15; // J/bit/m^alpha
static const double   PATH_LOSS_EXP     = 3.5;
static const uint32_t W_MIN             = 16;
static const uint32_t W_MAX             = 256;
static const double   BASE_TX_POWER_DBM = 0.0;
static const double   POWER_STEP_DBM    = 3.0;
static const double   BASE_RATE_KBPS    = 50.0;
static const double   RATE_SCALE        = 1.5;
static const uint32_t DQN_STATE_DIM     = 15;
static const uint32_t DQN_ACTION_DIM    = 81;  // 3^4

// ─── Cross-Layer State Vector ─────────────────────────────────────────────────
struct CrossLayerState {
  // PHY (4)
  double meanSNR, snrVariance, meanTxPower, meanResidualEnergy;
  // MAC (4)
  double meanContWin, meanCollRate, meanQueueLen, channelOccupancy;
  // NET (4)
  double meanHopCount, meanE2EDelay, aliveNodes, throughput;
  // TRANSPORT (3)
  double meanSendRate, rtt, commOverhead;
};

// ─── ACLIF Action Tuple ───────────────────────────────────────────────────────
struct ACLIFAction {
  int pwrDelta;    // {-1, 0, +1}
  int winDelta;    // {-1, 0, +1}
  int routeMode;  // {0=shortest, 1=energy-balanced, 2=load-balanced}
  int rateDelta;  // {-1, 0, +1}
};

// ─── Node Context ────────────────────────────────────────────────────────────
struct NodeCtx {
  double residualEnergy;
  double txPower;
  uint32_t contWin;
  double sendRate;
  uint32_t queueLen;
  double lastSNR;
  uint32_t hopCount;
  bool alive;
  double lastUpdateTime;
};

// ─── DQN Stub (forward pass placeholder) ─────────────────────────────────────
ACLIFAction DQNInference (const CrossLayerState& state, double epsilon) {
  // In production, this calls a Python subprocess via NS3 SystemThread.
  // Here we provide a rule-based approximation for simulation scaffolding.
  ACLIFAction a;
  // Power: increase if SNR is low
  if (state.meanSNR < 10.0)       a.pwrDelta = +1;
  else if (state.meanSNR > 25.0)  a.pwrDelta = -1;
  else                            a.pwrDelta =  0;
  // Window: expand if collision > 15%
  if (state.meanCollRate > 0.15)  a.winDelta = +1;
  else if (state.meanCollRate < 0.05) a.winDelta = -1;
  else                            a.winDelta =  0;
  // Route: energy-balanced if energy low
  if (state.meanResidualEnergy < 0.5)  a.routeMode = 1;
  else if (state.meanQueueLen > 10.0)  a.routeMode = 2;
  else                                 a.routeMode = 0;
  // Rate: back-off if RTT > 200ms
  if (state.rtt > 0.2)           a.rateDelta = -1;
  else if (state.rtt < 0.05)     a.rateDelta = +1;
  else                           a.rateDelta =  0;
  return a;
}

// ─── Staleness Filter ─────────────────────────────────────────────────────────
double StalenessWeight (double lastUpdateTime, double now,
                        double deltaMax, double tauS) {
  double excess = std::max(0.0, now - lastUpdateTime - deltaMax);
  return std::exp(-excess / tauS);
}

// ─── Energy Consumption ───────────────────────────────────────────────────────
double TxEnergy (uint32_t kBits, double distM) {
  return kBits * E_ELEC + kBits * E_AMP * std::pow(distM, PATH_LOSS_EXP);
}
double RxEnergy (uint32_t kBits) {
  return kBits * E_ELEC;
}

// ─── Collision Probability (CSMA/CA) ──────────────────────────────────────────
double CollisionRate (uint32_t W, uint32_t M) {
  if (M <= 1) return 0.0;
  return 1.0 - std::pow(1.0 - 1.0 / (double)W, (double)(M - 1));
}

// ─── Reward Function ──────────────────────────────────────────────────────────
double ComputeReward (double throughput, double delay,
                      double energy, double collision, double commOH) {
  const double w1=0.30, w2=0.25, w3=0.20, w4=0.15, w5=0.10;
  return w1*throughput - w2*delay - w3*energy - w4*collision - w5*commOH;
}

// ─── ICSA: State Aggregation ──────────────────────────────────────────────────
CrossLayerState ICSAAggregate (const std::vector<NodeCtx>& nodes,
                                double now,
                                double deltaMax = 1.0,
                                double tauS = 0.5) {
  CrossLayerState s = {};
  uint32_t valid = 0;
  double sumSNR=0, sumSNR2=0, sumPwr=0, sumEnergy=0;
  double sumW=0, sumColl=0, sumQ=0;
  double sumHop=0, sumDelay=0;
  uint32_t alive = 0;

  for (const auto& n : nodes) {
    if (!n.alive) continue;
    double w = StalenessWeight(n.lastUpdateTime, now, deltaMax, tauS);
    if (w < 0.5) continue;
    alive++;
    sumSNR   += n.lastSNR;
    sumSNR2  += n.lastSNR * n.lastSNR;
    sumPwr   += n.txPower;
    sumEnergy += n.residualEnergy;
    sumW     += n.contWin;
    double coll = CollisionRate(n.contWin, alive > 0 ? alive : 1);
    sumColl  += coll;
    sumQ     += n.queueLen;
    sumHop   += n.hopCount;
    valid++;
  }

  if (valid == 0) return s;
  double inv = 1.0 / valid;
  s.meanSNR           = sumSNR * inv;
  s.snrVariance       = (sumSNR2 * inv) - (s.meanSNR * s.meanSNR);
  s.meanTxPower       = sumPwr * inv;
  s.meanResidualEnergy = sumEnergy * inv;
  s.meanContWin       = sumW * inv;
  s.meanCollRate      = sumColl * inv;
  s.meanQueueLen      = sumQ * inv;
  s.aliveNodes        = (double)alive;
  s.meanHopCount      = sumHop * inv;
  return s;
}

// ─── APR: Apply Actions ───────────────────────────────────────────────────────
void APRReconfigure (std::vector<NodeCtx>& nodes, const ACLIFAction& a) {
  for (auto& n : nodes) {
    if (!n.alive) continue;
    // Power
    n.txPower = BASE_TX_POWER_DBM + POWER_STEP_DBM * a.pwrDelta;
    // Contention window (multiplicative)
    if (a.winDelta == +1) n.contWin = std::min((uint32_t)(n.contWin * 2), W_MAX);
    else if (a.winDelta == -1) n.contWin = std::max((uint32_t)(n.contWin / 2), W_MIN);
    // Send rate
    if (a.rateDelta == +1) n.sendRate = BASE_RATE_KBPS * RATE_SCALE;
    else if (a.rateDelta == -1) n.sendRate = BASE_RATE_KBPS / RATE_SCALE;
    else n.sendRate = BASE_RATE_KBPS;
  }
}

// ─── Main Simulation Entry ────────────────────────────────────────────────────
int main (int argc, char *argv[]) {
  // ── Command-line arguments ──
  uint32_t nNodes   = DEFAULT_N_NODES;
  double   pktRate  = PKT_RATE_DEFAULT;
  uint32_t runSeed  = 1;
  double   simTime  = SIM_TIME;
  std::string outputPrefix = "aclif_res10";

  CommandLine cmd;
  cmd.AddValue ("episode", "Sweep variable: episode", nNodes);
  cmd.AddValue ("pktRate",   "Packet generation rate (pkt/s)", pktRate);
  cmd.AddValue ("seed",      "RNG seed", runSeed);
  cmd.AddValue ("simTime",   "Simulation duration (s)", simTime);
  cmd.AddValue ("output",    "Output file prefix", outputPrefix);
  cmd.Parse (argc, argv);

  NS_LOG_UNCOND ("ACLIF Result 10 | nNodes=" << nNodes
                 << " pktRate=" << pktRate << " seed=" << runSeed);

  RngSeedManager::SetSeed (runSeed);
  RngSeedManager::SetRun  (runSeed);

  // ── Create nodes ──
  NodeContainer iotNodes;
  iotNodes.Create (nNodes);

  // ── Mobility ──
  MobilityHelper mobility;
  mobility.SetPositionAllocator (
    "ns3::RandomRectanglePositionAllocator",
    "X", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(AREA_SIDE) + "]"),
    "Y", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(AREA_SIDE) + "]")
  );
  // Static topology by default; swap to RandomWaypointMobilityModel for res12
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (iotNodes);

  // ── PHY / MAC: 802.15.4 CSMA/CA ──
  LrWpanHelper lrWpan;
  NetDeviceContainer devs = lrWpan.Install (iotNodes);
  lrWpan.AssociateToPan (devs, 0);

  // ── Internet stack ──
  InternetStackHelper internet;
  internet.Install (iotNodes);

  // ── Application: OnOff (Poisson traffic) ──
  uint16_t port = 9;
  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  // (address assignment omitted for 802.15.4 stub — full impl uses 6LoWPAN helper)

  // ── Flow monitor ──
  FlowMonitorHelper flowHelper;
  Ptr<FlowMonitor> monitor = flowHelper.InstallAll ();

  // ── ACLIF control loop state ──
  std::vector<NodeCtx> nodeCtxs (nNodes);
  for (uint32_t i = 0; i < nNodes; i++) {
    nodeCtxs[i].residualEnergy  = INIT_ENERGY;
    nodeCtxs[i].txPower         = BASE_TX_POWER_DBM;
    nodeCtxs[i].contWin         = W_MIN;
    nodeCtxs[i].sendRate        = BASE_RATE_KBPS;
    nodeCtxs[i].queueLen        = 0;
    nodeCtxs[i].lastSNR         = 15.0;
    nodeCtxs[i].hopCount        = 2;
    nodeCtxs[i].alive           = true;
    nodeCtxs[i].lastUpdateTime  = 0.0;
  }

  // ── Results output file ──
  std::ofstream outFile (outputPrefix + "_results.csv");
  outFile << "time_s,throughput_kbps,delay_ms,energy_J,collision_pct,reward\n";

  // ── Control loop (scheduled every CTRL_INTERVAL seconds) ──
  double prevReward = 0.0;
  uint32_t convCount = 0;
  const double DELTA_CONV = 0.01;
  const uint32_t K_CONV   = 10;

  Simulator::ScheduleWithContext (0, Seconds(0.0), [&]() {
    double now = Simulator::Now().GetSeconds();
    while (now < simTime) {
      // ICSA Phase
      CrossLayerState state = ICSAAggregate (nodeCtxs, now);
      // RLJO Phase
      double epsilon = std::max(0.1, std::exp(-0.01 * now));
      ACLIFAction action = DQNInference (state, epsilon);
      // APR Phase
      APRReconfigure (nodeCtxs, action);
      // Simulate one slot of energy drain
      for (auto& n : nodeCtxs) {
        if (!n.alive) continue;
        double eTx = TxEnergy (PKT_SIZE * 8, 50.0);
        double eRx = RxEnergy (PKT_SIZE * 8);
        n.residualEnergy -= (eTx + eRx);
        if (n.residualEnergy <= 0) { n.alive = false; n.residualEnergy = 0; }
        n.lastUpdateTime = now;
      }
      // Compute metrics
      double tput    = state.throughput;
      double delay   = state.meanE2EDelay;
      double energy  = INIT_ENERGY - state.meanResidualEnergy;
      double coll    = state.meanCollRate * 100.0;
      double commOH  = state.commOverhead;
      double reward  = ComputeReward (tput, delay, energy, coll, commOH);
      outFile << now << "," << tput << "," << delay << ","
              << energy << "," << coll << "," << reward << "\n";
      // Convergence check
      if (std::fabs(reward - prevReward) <= DELTA_CONV) {
        convCount++;
        if (convCount >= K_CONV) {
          NS_LOG_UNCOND ("ACLIF converged at t=" << now << "s");
          convCount = 0;
        }
      } else convCount = 0;
      prevReward = reward;
      now += CTRL_INTERVAL;
    }
  });

  Simulator::Stop (Seconds (simTime));
  Simulator::Run  ();

  // ── Flow statistics ──
  monitor->CheckForLostPackets ();
  Ptr<Ipv4FlowClassifier> classifier =
    DynamicCast<Ipv4FlowClassifier> (flowHelper.GetClassifier ());
  FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats ();
  double totalTput = 0, totalDelay = 0;
  uint64_t nFlows = 0;
  for (auto& kv : stats) {
    totalTput  += kv.second.rxBytes * 8.0 / simTime / 1000.0;
    if (kv.second.rxPackets > 0)
      totalDelay += kv.second.delaySum.GetSeconds () / kv.second.rxPackets;
    nFlows++;
  }
  NS_LOG_UNCOND ("Final Throughput : " << (nFlows ? totalTput/nFlows : 0) << " kbps");
  NS_LOG_UNCOND ("Final Avg Delay  : " << (nFlows ? totalDelay/nFlows*1000 : 0) << " ms");

  outFile.close ();
  Simulator::Destroy ();
  return 0;
}
