#include <cstdint>
#include <limits>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <functional>
#include <future>

using namespace std::chrono;
using Clock = high_resolution_clock;

template <typename Dur>
void prettyPrintDuration(Dur dur)
{
    std::cout << "Time:  ";

    auto h = duration_cast<hours>(dur);
    if (h.count()) { std::cout << h << " "; dur -= h; }

    auto m = duration_cast<minutes>(dur);
    if (m.count()) { std::cout << m << " "; dur -= m; }

    auto s = duration_cast<seconds>(dur);
    if (s.count()) { std::cout << s << " "; dur -= s; }

    auto ms = duration_cast<milliseconds>(dur);
    if (ms.count()) { std::cout << ms << " "; dur -= ms; }

    auto us = duration_cast<microseconds>(dur);
    if (us.count()) { std::cout << us << " "; dur -= us; }

    auto ns = duration_cast<nanoseconds>(dur);
    if (ns.count()) { std::cout << ns << " "; dur -= ns; }

    std::cout << std::endl;
}

template<typename Func>
auto measure(Func&& f) -> decltype(f())
{
    using namespace std::chrono;
    using clock = high_resolution_clock;

    auto start = clock::now();

    auto summarize = [&start]() {
        auto dt = Clock::now() - start;
        prettyPrintDuration(dt);
    };

    if constexpr (std::is_same_v<decltype(f()), void>) {
        f();
        summarize();
    }
    else {
        auto res = f();
        summarize();
        return res;
    }
}

using BigInt = uintmax_t;

std::string prettifyBigInt(BigInt val)
{
    std::string res = std::to_string(val);

    constexpr char FILLER = ' ';

    int acc = 0;
    for (int i = static_cast<int>(res.size() - 1); i >= 0; --i, ++acc) {
        if (acc == 3) {
            res.insert(static_cast<size_t>(i) + 1, 1, FILLER);
            ++i;
            acc = -1;
        }
    }

    return res;
}

struct LCG64 {
   using result_type = BigInt;
   uint64_t state;

   LCG64(uint64_t seed = 1) : state(seed) {}

   uint64_t operator()() {
      state = state * 6364136223846793005ULL + 1;
      return state; // full 64-bit value
   }
};

struct alignas(std::hardware_destructive_interference_size) ThreadData
{
   LCG64 gen{ std::random_device{}() };
   LCG64::result_type bits[std::hardware_destructive_interference_size/sizeof(LCG64::result_type) - sizeof(LCG64) / sizeof(LCG64::result_type)]{};
};

struct Experiment
{
    using Generator = LCG64;
    using BitsType = Generator::result_type;

    BigInt n = 0;
    BigInt heads = 0;
    BigInt tails = 0;

    static constexpr size_t BITS_COUNT = std::numeric_limits<BitsType>::digits;

    Experiment(BigInt n_)
        : n(n_) {
        if (n % 64 != 0) {
            std::cerr << "Error: n must be a multiple of 64, got " << n << "\n";
            std::abort();
        }
    }

    void spin() {
        const BigInt steps = n / BITS_COUNT;
        const size_t threadsCount = std::max(std::thread::hardware_concurrency(), 1u);
        const BigInt chunkSize = steps / threadsCount;

        if (chunkSize == 0) {
            Generator gen{ std::random_device{}() };
            for (BigInt i = 0; i < steps; ++i) {
                BitsType bits = gen();
                heads += std::popcount(bits);
            }
            tails = n - heads;
            return;
        }

        const auto thread = [this, chunkSize]() -> BigInt {
            ThreadData data;
            BigInt localHeads = 0;
            
            constexpr size_t s = std::size(data.bits);
            
            for (BigInt i = 0; i < chunkSize/s; ++i) {
                for (int j = 0; j < s; ++j) {
                    data.bits[j] = data.gen();
                }
                for (int j = 0; j < s; ++j) {
                    localHeads += std::popcount(data.bits[j]);
                }
            }
            return localHeads;
        };

        std::vector<std::future<BigInt>> threadHeads(threadsCount - 1);
        for (auto& f : threadHeads) {
            f = std::async(thread);
        }

        heads += thread();
        for (auto&& fut : threadHeads) {
            heads += fut.get();
        }

        tails = n - heads;
    }
};

int main()
{
   constexpr BigInt STEP = 4'294'967'296ll;
   constexpr BigInt AIM = 68'719'476'736ll;
   constexpr BigInt INTERATIONS = 16;

   constexpr BigInt START = STEP * (16 - INTERATIONS) + STEP;

   for (BigInt n = START; n <= AIM; n += STEP) {
        std::cout << prettifyBigInt(n) << " rounds" << std::endl;
        Experiment experiment(n);
        measure(std::bind(&Experiment::spin, &experiment));

        const double headsPercent = static_cast<double>(experiment.heads) / n * 100.0;
        const double tailsPercent = static_cast<double>(experiment.tails) / n * 100.0;

        std::cout << "Heads: " << prettifyBigInt(experiment.heads) << ", " << headsPercent << "%" << std::endl;
        std::cout << "Tails: " << prettifyBigInt(experiment.tails) << ", " << tailsPercent << "%" << std::endl << std::endl;
    };

    return 0;
}
