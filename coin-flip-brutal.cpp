#include <cstdint>
#include <limits>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cassert>
#include <functional>

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

template<typename Func>
void spin(BigInt n, Func&& f)
{
    for (BigInt i = 0; i < n; ++i) {
        f();
    }
}

struct Experiment
{
    Experiment()
    {
        regenBits();
    }

    using Generator = std::mt19937;
    using BitsType = Generator::result_type;

    BigInt heads = 0;
    BigInt tails = 0;

    Generator gen {std::random_device{}()};
    BitsType bits{};
    int bitsCounter = 0;

    void tossCoin() {
        if (yieldBit()) {
            ++heads;
        } else {
            ++tails;
        }
    }

    bool yieldBit() {
        if (bitsCounter >= std::numeric_limits<BitsType>::digits) {
            regenBits();
        }

        return (bits >> bitsCounter++) & 1;
    }

    void regenBits() {
        bitsCounter = 0;
        bits = gen();
    }
};

int main()
{
    for (BigInt n : {
        256ll,
        65536ll,
        4294967296ll,
        /*4294967296ll * 2,
        4294967296ll * 3,
        4294967296ll * 4,*/
    }) {
        std::cout << prettifyBigInt(n) << " rounds" << std::endl;
        Experiment experiment;
        measure([&experiment, n]() {
            spin(n,
                std::bind(&Experiment::tossCoin, &experiment)
            );
        });

        const double headsPercent = static_cast<double>(experiment.heads) / n * 100.0;
        const double tailsPercent = static_cast<double>(experiment.tails) / n * 100.0;

        std::cout << "Heads: " << prettifyBigInt(experiment.heads) << ", " << headsPercent << "%" << std::endl;
        std::cout << "Tails: " << prettifyBigInt(experiment.tails) << ", " << tailsPercent << "%" << std::endl << std::endl;
    };

    return 0;
}
