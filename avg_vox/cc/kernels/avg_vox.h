#ifndef KERNEL_AVG_VOX_H_
#define KERNEL_AVG_VOX_H_

namespace tensorflow {
namespace functor {

template <typename Device>
struct AvgVoxForwardFunctor {
  void operator()(const Device& d);
};

}  // namespace functor
}  // namespace tensorflow

#endif // KERNEL_AVG_VOX_H_
