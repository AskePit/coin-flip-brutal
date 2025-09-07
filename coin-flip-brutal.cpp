#include <cstdint>
#include <limits>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <functional>
#include <future>
#include <new>
#include <immintrin.h>

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

   __forceinline uint64_t operator()() {
       state = state * 6364136223846793005ULL + 1;
       return state; // full 64-bit value
   }
};

struct XorShift64 {
   using result_type = BigInt;
   uint64_t state;

   XorShift64(uint64_t seed = 88172645463325252ULL) : state(seed) {}

   uint64_t operator()() {
      state ^= state >> 12;
      state ^= state << 25;
      state ^= state >> 27;
      return state; // full 64-bit random integer
   }
};

struct HwRandom64 {
   using result_type = BigInt;

   HwRandom64(uint64_t seed = 88172645463325252ULL) { (void)seed; }

   uint64_t operator()() {
      uint64_t val;
      if (_rdrand64_step(&val)) {
         return val;
      }
      throw std::runtime_error("RDRAND failed");
   }
};

struct alignas(std::hardware_destructive_interference_size) IntFuture
{
    std::future<BigInt> val;
};

static constexpr size_t SIMD_BATCH = 32;
static_assert(SIMD_BATCH % 4 == 0, "shouldda be multiple of four!");

static uint64_t scalarPopcount(const uint64_t* data) {
   uint64_t sum = 0;
   for (size_t i = 0; i < SIMD_BATCH; i++)
      sum += std::popcount(data[i]);
   return sum;
}

static uint64_t avx2Popcount(const uint64_t* data) {
   static const __m256i lut = _mm256_setr_epi8(
      0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
      0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
   );

   __m256i total = _mm256_setzero_si256();
   const __m256i zeroF = _mm256_set1_epi8(0x0F);

   size_t i = 0;
   for (; i < SIMD_BATCH; i += 4) {
      __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(&data[i]));

      __m256i lo = _mm256_and_si256(v, zeroF);
      __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), zeroF);

      __m256i cnt = _mm256_add_epi8(
         _mm256_shuffle_epi8(lut, lo),
         _mm256_shuffle_epi8(lut, hi)
      );

      total = _mm256_add_epi64(total, _mm256_sad_epu8(cnt, _mm256_setzero_si256()));
   }

   __m128i low128 = _mm256_castsi256_si128(total);
   __m128i high128 = _mm256_extracti128_si256(total, 1);
   __m128i sum128 = _mm_add_epi64(low128, high128);
   uint64_t result = _mm_cvtsi128_si64(sum128) + static_cast<uint64_t>(_mm_extract_epi64(sum128, 1));

   return result;
}


uint64_t avx512Popcount(const uint64_t* data) {
   __m512i acc = _mm512_setzero_si512();
   size_t i = 0;
   for (; i < SIMD_BATCH; i += 8) {
      __m512i v = _mm512_loadu_si512(&data[i]);
      __m512i pc = _mm512_popcnt_epi64(v);
      acc = _mm512_add_epi64(acc, pc);
   }
   uint64_t result = _mm512_reduce_add_epi64(acc);
   return result;
}

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
        if (n % SIMD_BATCH * 64 != 0) {
            std::cerr << "Error: n must be a multiple of 64, got " << n << "\n";
            std::abort();
        }
    }

    void spin() {
        const BigInt steps = n / BITS_COUNT;
        const size_t threadsCount = std::max(std::thread::hardware_concurrency()*64, 1u);
        const BigInt chunkSize = steps / threadsCount;

        const auto thread = [this, chunkSize]() -> BigInt {
            Generator gen{ std::random_device{}() };
            BigInt localHeads = 0;
            alignas(std::hardware_destructive_interference_size) BitsType bits[SIMD_BATCH];

            for (BigInt i = 0; i < chunkSize / SIMD_BATCH; ++i) {
                for (int j = 0; j < SIMD_BATCH; ++j) {
                    bits[j] = gen();
                }
                localHeads += avx2Popcount(&bits[0]);
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

class Divider
{
public:
    Divider(BigInt nom_, BigInt denom_)
        : nom(nom_)
        , denom(denom_)
    {}

    int operator()() {
        if (nom == 0) {
            return -1;
        }

        nom *= 10;
        if (nom < denom) {
            return 0;
        }

        BigInt acc = denom;
        while (acc <= nom) {
            acc += denom;
        }
        acc -= denom;

        int digit = static_cast<int>(acc / denom);
        nom -= acc;

        return digit;
    }

private:
    BigInt nom{};
    BigInt denom{};
};

class Decimal
{
public:
    Decimal(size_t digitsAfterComma_, uint64_t underlying_ = 0)
       : digitsAfterComma(digitsAfterComma_)
       , underlying(underlying_)
    {
        scale = static_cast<size_t>(std::pow(10, digitsAfterComma));
    }

    size_t getScale() const {
        return scale;
    }

    void addFracts(size_t fracts) {
       underlying += fracts;
    }

    bool isHalf() const {
        return underlying == 5 * (scale/10);
    }

    std::string toString() const {
        std::string res("0.");
        res += std::to_string(underlying);
        while (res[res.size() - 1] == '0') {
            res.resize(res.size() - 1);
        }

        return res;
    }

private:
    size_t digitsAfterComma{};
    size_t scale{};
    uint64_t underlying{};
};

Decimal calcPercent(BigInt nom, BigInt denom, size_t digitsAfterComma) {
    Decimal res(digitsAfterComma);
    Divider divider(nom, denom);

    size_t scale = res.getScale() / 10;

    while (scale != 0) {
        int digit = divider();
        if (digit < 0) {
            break;
        }
        res.addFracts(digit * scale);
        scale /= 10;
    }

    // rounding
    int digit = divider();
    if (digit >= 5) {
        res.addFracts(1);
    }

    return res;
}

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

        const Decimal headsPercent = calcPercent(experiment.heads, n, 10);
        const Decimal tailsPercent = calcPercent(experiment.tails, n, 10);

        std::cout << "Heads: " << prettifyBigInt(experiment.heads) << ", " << headsPercent.toString() << std::endl;
        std::cout << "Tails: " << prettifyBigInt(experiment.tails) << ", " << tailsPercent.toString() << std::endl << std::endl;
    };

    return 0;
}
