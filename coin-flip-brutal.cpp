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

template<std::unsigned_integral T, typename Func>
void spin(Func&& f)
{
    for (T i = 0; ; ++i) {
        f();
        if (i == std::numeric_limits<T>::max()) break;
    }
}

template<typename BigInt>
struct Experiment
{
    std::mt19937 gen {std::random_device{}()};
    std::uniform_int_distribution<int> uni {0, 1};
    BigInt heads = 0;
    BigInt tails = 0;

    void tossCoin() {
        int res = uni(gen);
        if (res == 1) {
            ++heads;
        }
        else if (res == 0) {
            ++tails;
        }
        else {
            assert(false);
        }
    }
};

int main()
{
    using BigInt = uint16_t;
    Experiment<BigInt> experiment;

    measure([&experiment](){
        spin<BigInt>(
            std::bind(&Experiment<BigInt>::tossCoin, &experiment)
        );
    });

    const double headsPercent = experiment.heads / (static_cast<double>(std::numeric_limits<BigInt>::max()) + 1.0) * 100.0;
    const double tailsPercent = experiment.tails / (static_cast<double>(std::numeric_limits<BigInt>::max()) + 1.0) * 100.0;

    std::cout << "Heads: " << +experiment.heads << ", " << headsPercent << "%" << std::endl;
    std::cout << "Tails: " << +experiment.tails << ", " << tailsPercent << "%" << std::endl;

    return 0;
}
