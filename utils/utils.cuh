#ifndef _UTILS_H
#define _UTILS_H

#include <algorithm>
#include <cmath>

#define MAXIMUM_THREADS 512

inline int optimal_num_threads(int work_size) {
  const int pow_2 = std::log2(static_cast<double>(work_size));
  return std::max(std::min(1 << pow_2, MAXIMUM_THREADS), 1);
}

#endif // _UTILS_H