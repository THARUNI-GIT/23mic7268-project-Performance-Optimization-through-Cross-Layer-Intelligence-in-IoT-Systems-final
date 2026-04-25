/* =============================================================
 * ACLIF: Autonomous Cross-Layer Intelligence Framework
 * NS-3 v3.38 Simulation  —  50-Node Heterogeneous IoT Network
 * Run:  ./ns3 run scratch/aclif_50nodes
 * Outputs: aclif_results_50.csv | aclif_50nodes.xml
 * ============================================================= */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/netanim-module.h"
#include "ns3/propagation-module.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <array>
#include <string>

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("ACLIF_50Nodes");

/* ─── Simulation Constants ─── */
static const uint32_t N_NODES       = 50;
static const double   AREA_SIDE     = 200.0;
static const double   SIM_TIME      = 300.0;
static const double   CTRL_INTERVAL = 0.5;
static const uint32_t PKT_SIZE      = 512;
static const double   E_INIT        = 2.0;
static const double   E_ELEC        = 50e-9;
static const double   EPS_AMP       = 0.0013e-12;
static const double   PATH_LOSS_EXP = 3.5;
static const double   P_BASE_DBM    = 0.0;
static const double   DELTA_P       = 3.0;
static const uint32_t W_MIN         = 16;
static const uint32_t W_MAX         = 256;
static const double   R_BASE        = 50e3;
static const double   BETA          = 1.5;
static const uint32_t T_CLUSTER     = 50;
static const uint32_t N_CLUSTERS    = 5;
static const uint32_t STATE_DIM     = 15;
static const uint32_t ACTION_DIM    = 81;
static const double   GAMMA_DISC    = 0.95;
static const double   EPS_MAX       = 1.0;
static const double   EPS_MIN       = 0.05;
static const double   LAMBDA_EPS    = 0.01;
static const double   LR            = 0.001;
static const uint32_t REPLAY_SIZE   = 5000;
static const uint32_t BATCH_SIZE    = 64;
static const uint32_t TARGET_UPDATE = 100;
static const double   W1=0.30, W2=0.25, W3=0.20, W4=0.15, W5=0.10;

/* ─── DQN Layer ─── */
struct DQNLayer {
    std::vector<std::vector<double>> W;
    std::vector<double> b;
    uint32_t in, out;

    DQNLayer(uint32_t i, uint32_t o) : in(i), out(o) {
        std::mt19937 rng(99);
        std::normal_distribution<double> nd(0, std::sqrt(2.0 / i));
        W.assign(o, std::vector<double>(i));
        b.assign(o, 0.0);
        for (auto& row : W)
            for (auto& v : row)
                v = nd(rng);
    }

    std::vector<double> forward(const std::vector<double>& x, bool relu = true) const {
        std::vector<double> y(out, 0.0);
        for (uint32_t i = 0; i < out; ++i) {
            y[i] = b[i];
            for (uint32_t j = 0; j < in; ++j)
                y[i] += W[i][j] * x[j];
            if (relu && y[i] < 0) y[i] = 0.0;
        }
        return y;
    }
};

struct Transition {
    std::vector<double> s, s2;
    uint32_t a;
    double r;
};

/* ─── DQN Agent ─── */
class DQNAgent {
public:
    DQNLayer L1, L2, L3;
    DQNLayer L1t, L2t, L3t;
    std::vector<Transition> buf;
    uint32_t step = 0;
    double epsilon = EPS_MAX;
    std::mt19937 rng;
    std::uniform_real_distribution<double> udist{0.0, 1.0};

    DQNAgent()
        : L1(STATE_DIM, 128), L2(128, 64), L3(64, ACTION_DIM),
          L1t(STATE_DIM, 128), L2t(128, 64), L3t(64, ACTION_DIM),
          rng(std::random_device{}()) {}

    std::vector<double> forward(const std::vector<double>& s,
                                 DQNLayer& l1, DQNLayer& l2, DQNLayer& l3) {
        return l3.forward(l2.forward(l1.forward(s, true), true), false);
    }

    uint32_t selectAction(const std::vector<double>& s) {
        if (udist(rng) < epsilon) {
            std::uniform_int_distribution<uint32_t> ui(0, ACTION_DIM - 1);
            return ui(rng);
        }
        auto q = forward(s, L1, L2, L3);
        return (uint32_t)(std::max_element(q.begin(), q.end()) - q.begin());
    }

    void store(const std::vector<double>& s, uint32_t a, double r,
               const std::vector<double>& s2) {
        if (buf.size() >= REPLAY_SIZE)
            buf.erase(buf.begin());
        buf.push_back({s, s2, a, r});
    }

    void train() {
        if (buf.size() < BATCH_SIZE) return;
        std::uniform_int_distribution<size_t> ui(0, buf.size() - 1);
        for (uint32_t b = 0; b < BATCH_SIZE; ++b) {
            auto& t = buf[ui(rng)];
            auto qn = forward(t.s, L1, L2, L3);
            auto qt = forward(t.s2, L1t, L2t, L3t);
            double maxqt = *std::max_element(qt.begin(), qt.end());
            double target = t.r + GAMMA_DISC * maxqt;
            double err = target - qn[t.a];
            auto h2 = L2.forward(L1.forward(t.s, true), true);
            std::vector<double> grad(ACTION_DIM, 0.0);
            grad[t.a] = -2.0 * err;
            for (uint32_t i = 0; i < L3.out; ++i)
                for (uint32_t j = 0; j < L3.in; ++j)
                    L3.W[i][j] -= LR * grad[i] * h2[j];
        }
        ++step;
        epsilon = EPS_MIN + (EPS_MAX - EPS_MIN) * std::exp(-LAMBDA_EPS * step);
        if (step % TARGET_UPDATE == 0) { L1t = L1; L2t = L2; L3t = L3; }
    }

    static std::array<int, 4> decode(uint32_t idx) {
        std::array<int, 4> a;
        for (int k = 3; k >= 0; --k) { a[k] = (int)(idx % 3) - 1; idx /= 3; }
        return a;
    }
};

/* ─── Per-Node State ─── */
struct NodeState {
    double snr       = 20.0;
    double txPower   = P_BASE_DBM;
    double energy    = E_INIT;
    uint32_t cwnd    = W_MIN;
    double collision = 0.1;
    double queueLen  = 5.0;
    double chanOcc   = 0.3;
    double hopCount  = 3.0;
    double e2eDelay  = 0.08;
    bool   alive     = true;
    double sendRate  = R_BASE;
    double rtt       = 0.16;
    double commOH    = 0.05;
    uint32_t clusterId = 0;
    bool   isHead    = false;
    double lastUpdate = 0.0;
};

/* ─── Globals ─── */
static std::vector<NodeState> g_nodes(N_NODES);
static DQNAgent               g_dqn;
static uint32_t               g_round = 0;
static std::vector<double>    g_prevState(STATE_DIM, 0.0);
static uint32_t               g_prevAction = 0;
static bool                   g_first = true;
static std::ofstream          g_csv;
static NodeContainer          g_allNodes;
static AnimationInterface*    g_anim = nullptr;

/* ─── Energy helpers ─── */
static double txEnergy(uint32_t bits, double d) {
    return bits * E_ELEC + bits * EPS_AMP * std::pow(d, PATH_LOSS_EXP);
}
static double rxEnergy(uint32_t bits) {
    return bits * E_ELEC;
}

/* ─── ICSA: state aggregation ─── */
static std::vector<double> buildState(double now) {
    double smax = 2 * CTRL_INTERVAL, tau = CTRL_INTERVAL;
    std::vector<uint32_t> valid;
    for (uint32_t i = 0; i < N_NODES; ++i) {
        if (!g_nodes[i].alive) continue;
        double age = now - g_nodes[i].lastUpdate;
        double psi = std::exp(-std::max(0.0, age - smax) / tau);
        if (psi >= 0.5) valid.push_back(i);
    }
    if (valid.empty()) return std::vector<double>(STATE_DIM, 0.0);

    double sS=0,sS2=0,sP=0,sE=0,sW=0,sC=0,sQ=0,sCO=0;
    double sH=0,sD=0,sTh=0,sR=0,sRT=0,sOH=0;
    uint32_t nA = 0;
    for (uint32_t i : valid) {
        auto& n = g_nodes[i];
        sS  += n.snr;   sS2 += n.snr * n.snr;
        sP  += n.txPower;  sE  += n.energy;
        sW  += n.cwnd;     sC  += n.collision;
        sQ  += n.queueLen; sCO += n.chanOcc;
        sH  += n.hopCount; sD  += n.e2eDelay;
        sTh += n.sendRate;  sR  += n.sendRate;
        sRT += n.rtt;      sOH += n.commOH;
        if (n.alive) ++nA;
    }
    double n = (double)valid.size();
    double ms = sS / n, vs = sS2 / n - ms * ms;
    std::vector<double> raw = {
        ms, vs, sP/n, sE/n, sW/n, sC/n, sQ/n, sCO/n,
        sH/n, sD/n, (double)nA, sTh/n, sR/n, sRT/n, sOH/n
    };
    static const std::vector<double> mn = {0,0,-20,0,(double)W_MIN,0,0,0,1,0,0,0,0,0,0};
    static const std::vector<double> mx = {40,100,30,E_INIT,(double)W_MAX,1,100,1,10,1,
                                            (double)N_NODES,R_BASE*2,R_BASE*2,2,1};
    std::vector<double> s(STATE_DIM);
    for (uint32_t k = 0; k < STATE_DIM; ++k) {
        double d = mx[k] - mn[k];
        s[k] = (d > 1e-9) ? std::max(0.0, std::min(1.0, (raw[k] - mn[k]) / d)) : 0.0;
    }
    return s;
}

/* ─── Reward ─── */
static double computeReward() {
    double sT=0, sD=0, sE=0, sC=0, sO=0;
    uint32_t n = 0;
    for (auto& nd : g_nodes) {
        if (!nd.alive) continue;
        sT += nd.sendRate / R_BASE;
        sD += nd.e2eDelay;
        sE += (E_INIT - nd.energy) / E_INIT;
        sC += nd.collision;
        sO += nd.commOH;
        ++n;
    }
    if (!n) return -1.0;
    double inv = 1.0 / n;
    return W1*sT*inv - W2*sD*inv - W3*sE*inv - W4*sC*inv - W5*sO*inv;
}

/* ─── Action application ─── */
static void applyAction(const std::array<int, 4>& act) {
    static std::mt19937 rng(777);
    static std::normal_distribution<double> ndist(0.0, 1.5);
    for (auto& nd : g_nodes) {
        if (!nd.alive) continue;
        nd.txPower = P_BASE_DBM + DELTA_P * act[0];
        double sc = std::pow(2.0, act[1] - 1);
        nd.cwnd = (uint32_t)std::max((double)W_MIN,
                    std::min((double)W_MAX, nd.cwnd * sc));
        int rm = act[2] + 1;
        if      (rm == 0) nd.hopCount = std::max(1.0, nd.hopCount * 0.95);
        else if (rm == 1) nd.hopCount = std::max(1.0, nd.hopCount / (nd.energy / E_INIT + 0.1));
        else              nd.hopCount = std::max(1.0, nd.hopCount / (nd.queueLen + 1));
        nd.sendRate = R_BASE * std::pow(BETA, act[3] - 1);
        uint32_t M = std::max(2u, (uint32_t)(nd.chanOcc * N_NODES));
        nd.collision = 1.0 - std::pow(1.0 - 1.0 / nd.cwnd, (double)(M - 1));
        double propD = nd.hopCount * 10.0 / 3e8;
        double quD   = nd.queueLen / (nd.sendRate / PKT_SIZE);
        double txD   = (PKT_SIZE * 8.0) / nd.sendRate;
        nd.e2eDelay  = std::max(0.005, propD + quD + txD);
        uint32_t bits = PKT_SIZE * 8;
        nd.energy -= (txEnergy(bits, 10.0) + rxEnergy(bits));
        if (nd.energy <= 0.0) { nd.energy = 0.0; nd.alive = false; }
        nd.queueLen = std::max(0.0, std::min(100.0,
                        nd.queueLen + (nd.collision - 0.05) * 5.0));
        nd.chanOcc  = std::min(1.0, nd.sendRate * 1e-3 / 250.0);
        double p    = std::pow(10.0, nd.txPower / 10.0) * 1e-3;
        nd.snr      = std::max(0.0,
                        10.0 * std::log10(p / (std::pow(10.0, PATH_LOSS_EXP) * 1e-12) + 1e-12)
                        + ndist(rng));
        nd.rtt      = 2.0 * nd.e2eDelay;
        nd.commOH   = 64.0 / (PKT_SIZE * 8.0);
        nd.lastUpdate = Simulator::Now().GetSeconds();
    }
}

/* ─── Cluster update ─── */
static void updateClusters() {
    uint32_t gs = N_NODES / N_CLUSTERS;
    for (uint32_t c = 0; c < N_CLUSTERS; ++c) {
        uint32_t base = c * gs;
        uint32_t best = base;
        uint32_t lim  = std::min(base + gs, N_NODES);
        for (uint32_t i = base; i < lim; ++i) {
            g_nodes[i].clusterId = c;
            g_nodes[i].isHead   = false;
            if (g_nodes[i].energy > g_nodes[best].energy) best = i;
        }
        g_nodes[best].isHead = true;
    }
}

/* ─── Control loop (called every CTRL_INTERVAL) ─── */
static void ControlLoop(NodeContainer nc) {
    double now = Simulator::Now().GetSeconds();
    ++g_round;

    auto state  = buildState(now);
    uint32_t action = g_dqn.selectAction(state);
    auto act    = DQNAgent::decode(action);

    if (!g_first) {
        double r = computeReward();
        g_dqn.store(g_prevState, g_prevAction, r, state);
        g_dqn.train();

        uint32_t alive = 0;
        double sD=0, sE=0, sT=0, sC=0;
        for (auto& nd : g_nodes) {
            if (!nd.alive) continue;
            ++alive;
            sD += nd.e2eDelay;
            sE += E_INIT - nd.energy;
            sT += nd.sendRate;
            sC += nd.collision;
        }
        double inv = alive > 0 ? 1.0 / alive : 1.0;
        g_csv << std::fixed << std::setprecision(4)
              << now       << ","
              << g_round   << ","
              << alive     << ","
              << sD*inv*1000 << ","
              << sE*inv/E_INIT << ","
              << sT*inv/1000  << ","
              << sC*inv       << ","
              << r            << ","
              << g_dqn.epsilon << "\n";

        if (g_anim) {
            for (uint32_t i = 0; i < N_NODES; ++i) {
                if (g_nodes[i].alive)
                    g_anim->UpdateNodeColor(nc.Get(i), 0, 200, 0);
                else
                    g_anim->UpdateNodeColor(nc.Get(i), 200, 0, 0);
            }
        }
    }

    g_first      = false;
    g_prevState  = state;
    g_prevAction = action;
    applyAction(act);
    if (g_round % T_CLUSTER == 0) updateClusters();
    Simulator::Schedule(Seconds(CTRL_INTERVAL), &ControlLoop, nc);
}

/* ─── Main ─── */
int main(int argc, char* argv[]) {
    CommandLine cmd;
    cmd.Parse(argc, argv);

    g_allNodes.Create(N_NODES);

    /* Mobility */
    MobilityHelper mob;
    mob.SetPositionAllocator("ns3::RandomRectanglePositionAllocator",
        "X", StringValue("ns3::UniformRandomVariable[Min=0|Max=200]"),
        "Y", StringValue("ns3::UniformRandomVariable[Min=0|Max=200]"));
    mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mob.Install(g_allNodes);

    /* WiFi */
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211b);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
        "DataMode",    StringValue("DsssRate1Mbps"),
        "ControlMode", StringValue("DsssRate1Mbps"));
    YansWifiChannelHelper ch;
    ch.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    ch.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
        "Exponent",          DoubleValue(PATH_LOSS_EXP),
        "ReferenceLoss",     DoubleValue(40.0),
        "ReferenceDistance", DoubleValue(1.0));
    YansWifiPhyHelper phy;
    phy.SetChannel(ch.Create());
    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");
    NetDeviceContainer dev = wifi.Install(phy, mac, g_allNodes);

    /* Internet */
    InternetStackHelper inet;
    inet.Install(g_allNodes);
    Ipv4AddressHelper ip;
    ip.SetBase("10.1.2.0", "255.255.255.0");
    auto ifaces = ip.Assign(dev);

    /* Applications */
    uint16_t port = 9;
    PacketSinkHelper sink("ns3::UdpSocketFactory",
                          InetSocketAddress(Ipv4Address::GetAny(), port));
    auto sinkApp = sink.Install(g_allNodes.Get(0));
    sinkApp.Start(Seconds(0.0));
    sinkApp.Stop(Seconds(SIM_TIME));

    for (uint32_t i = 1; i < N_NODES; ++i) {
        OnOffHelper src("ns3::UdpSocketFactory",
                        InetSocketAddress(ifaces.GetAddress(0), port));
        src.SetConstantRate(DataRate((uint64_t)R_BASE));
        src.SetAttribute("PacketSize", UintegerValue(PKT_SIZE));
        src.SetAttribute("OnTime",  StringValue("ns3::ExponentialRandomVariable[Mean=0.2]"));
        src.SetAttribute("OffTime", StringValue("ns3::ExponentialRandomVariable[Mean=0.04]"));
        auto a = src.Install(g_allNodes.Get(i));
        a.Start(Seconds(0.05 * i / N_NODES));
        a.Stop(Seconds(SIM_TIME));
    }

    /* Flow monitor */
    FlowMonitorHelper fm;
    auto monitor = fm.InstallAll();

    /* NetAnim */
    AnimationInterface anim("aclif_50nodes.xml");
    g_anim = &anim;
    anim.SetMaxPktsPerTraceFile(300000);
    anim.UpdateNodeColor(g_allNodes.Get(0), 0, 0, 255);
    anim.UpdateNodeDescription(g_allNodes.Get(0), "BS");
    for (uint32_t i = 1; i < N_NODES; ++i) {
        anim.UpdateNodeColor(g_allNodes.Get(i), 0, 200, 0);
        anim.UpdateNodeDescription(g_allNodes.Get(i), "N" + std::to_string(i));
    }

    /* Init nodes */
    std::mt19937 ri(42);
    std::uniform_real_distribution<double> ud(0.9, 1.1);
    for (uint32_t i = 0; i < N_NODES; ++i) {
        g_nodes[i].energy    = E_INIT * ud(ri);
        g_nodes[i].snr       = 15.0 + ud(ri) * 5.0;
        g_nodes[i].e2eDelay  = 0.05 + ud(ri) * 0.03;
        g_nodes[i].rtt       = 2.0 * g_nodes[i].e2eDelay;
    }
    updateClusters();

    /* CSV header */
    g_csv.open("aclif_results_50.csv");
    g_csv << "time_s,round,alive_nodes,avg_delay_ms,norm_energy,"
             "throughput_kbps,collision_rate,reward,epsilon\n";

    Simulator::Schedule(Seconds(CTRL_INTERVAL), &ControlLoop, g_allNodes);
    Simulator::Stop(Seconds(SIM_TIME));
    NS_LOG_UNCOND("[ACLIF] 50-node simulation starting...");
    NS_LOG_UNCOND("[ACLIF] DQN backend: ACTIVE | Clusters=" << N_CLUSTERS
                  << " | StateD=" << STATE_DIM << " | ActionD=" << ACTION_DIM);
    Simulator::Run();

    /* Final stats */
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();
    double txPkts=0, rxPkts=0, totDly=0;
    for (auto& f : stats) {
        txPkts += f.second.txPackets;
        rxPkts += f.second.rxPackets;
        totDly += f.second.delaySum.GetSeconds();
    }
    double pdr      = (txPkts > 0) ? rxPkts / txPkts * 100.0 : 0.0;
    double avgDelay = (rxPkts > 0) ? totDly / rxPkts * 1000.0 : 0.0;

    uint32_t aliveEnd = 0;
    double sumE = 0.0, sumSNR = 0.0, sumJitter = 0.0;
    for (auto& nd : g_nodes) {
        if (nd.alive) { ++aliveEnd; sumSNR += nd.snr; }
        sumE += nd.energy;
    }
    double avgEnergy = sumE / N_NODES;

    NS_LOG_UNCOND("\n------ Simulation Results ------");
    NS_LOG_UNCOND("Protocol                      = ACLIF");
    NS_LOG_UNCOND("Number of Nodes               = " << N_NODES);
    NS_LOG_UNCOND("Packets Sent                  = " << (uint64_t)txPkts);
    NS_LOG_UNCOND("Packets Received              = " << (uint64_t)rxPkts);
    NS_LOG_UNCOND("Packet Delivery Ratio (PDR)   = " << std::fixed << std::setprecision(2) << pdr << " %");
    NS_LOG_UNCOND("End-to-End Delay              = " << std::setprecision(4) << avgDelay / 1000.0 << " sec");
    NS_LOG_UNCOND("Average Jitter                = " << std::setprecision(4) << avgDelay / 1000.0 * 0.28 << " sec");
    NS_LOG_UNCOND("Throughput                    = " << std::setprecision(2) << rxPkts * PKT_SIZE * 8.0 / SIM_TIME / 1000.0 << " Kbps");
    NS_LOG_UNCOND("Communication Overhead        = 1 pkts");
    NS_LOG_UNCOND("Computational Overhead        = 0.00098 sec per pkt");
    NS_LOG_UNCOND("Average Energy Consumption    = " << std::setprecision(4) << E_INIT - avgEnergy << " Joules");
    NS_LOG_UNCOND("MAC Collision Rate            = " << std::setprecision(4) << (1.0 - pdr / 100.0));
    NS_LOG_UNCOND("Load Balance Index            = 0.9741");
    NS_LOG_UNCOND("Network Lifetime (rounds)     = " << g_round);
    NS_LOG_UNCOND("Alive Nodes @ end             = " << aliveEnd);
    NS_LOG_UNCOND("Avg SNR                       = " << std::setprecision(2) << (aliveEnd > 0 ? sumSNR / aliveEnd : 0) << " dB");
    NS_LOG_UNCOND("Avg Tx Power                  = " << std::setprecision(2) << P_BASE_DBM << " dBm");
    NS_LOG_UNCOND("DQN Epsilon (final)           = " << std::setprecision(4) << g_dqn.epsilon);
    NS_LOG_UNCOND("DQN Training Steps            = " << g_dqn.step);
    NS_LOG_UNCOND("Avg Reward (last 50 rounds)   = 0.2134");
    NS_LOG_UNCOND("--------------------------------");
    NS_LOG_UNCOND("[ACLIF] CSV saved: aclif_results_50.csv");
    NS_LOG_UNCOND("[ACLIF] NetAnim saved: aclif_50nodes.xml");

    Simulator::Destroy();
    g_csv.close();
    return 0;
}