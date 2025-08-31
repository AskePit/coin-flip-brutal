#include <cstdint>
#include <limits>
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace std::chrono;
using Clock = high_resolution_clock;

void prettyPrintDuration(Clock::duration dur)
{
    if (duration_cast<hours>(dur).count() >= 1) {
        std::cout << duration_cast<hours>(dur);
    } else if (duration_cast<minutes>(dur).count() >= 1) {
        std::cout << duration_cast<minutes>(dur);
    } else if (duration_cast<seconds>(dur).count() >= 1) {
        std::cout << duration_cast<seconds>(dur);
    } else if (duration_cast<milliseconds>(dur).count() >= 1) {
        std::cout << duration_cast<milliseconds>(dur);
    } else if (duration_cast<microseconds>(dur).count() >= 1) {
        std::cout << duration_cast<microseconds>(dur);
    } else {
        std::cout << dur;
    }
    std::cout << std::endl;
}

template<typename Func>
void measure(Func f)
{
    using namespace std::chrono;
    using clock = high_resolution_clock;
    auto start = clock::now();

    f();
    auto dt = Clock::now() - start;
    prettyPrintDuration(dt);
}

template<std::unsigned_integral T>
void spin()
{
    for (T i = 0; ; ++i) {
        std::cout << std::dec << +i << " "; // `+` promotes char to integer 
        if (i == std::numeric_limits<T>::max()) break;
    }
    std::cout << std::endl;
}

int main()
{
    measure(spin<uint8_t>);
    return 0;
}
