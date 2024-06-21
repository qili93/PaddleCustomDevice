// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include <iostream>
#include <numeric>

#include "funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void doFloorDivTensor(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      phi::DenseTensor* out) {
  VLOG(4) << "call tecodnn floordivide";

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> y_dims = phi::vectorize<int>(y.dims());
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());

  // FIXME(huangzhen): the dims of scalar is 0 in static graph mode which is not
  // supported, so we change dims from [] to [1]`
  if (y_dims.empty()) {
    PADDLE_ENFORCE_GT(
        y.numel(),
        0,
        phi::errors::InvalidArgument("DenseTensor's numel can't be empty."));
    y_dims.push_back(1);
  }

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_dims, x.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t y_Desc = sdaa_ops::GetTecodnnTensorDesc(
      y_dims, y.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t out_Desc = sdaa_ops::GetTecodnnTensorDesc(
      out_dims, out->dtype(), TensorFormat::Undefined);

  TECODNN_CHECK(tecodnnFloorDivide(tecodnnHandle,
                                   x_Desc,
                                   x.data(),
                                   y_Desc,
                                   y.data(),
                                   out_Desc,
                                   out->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
}

template <typename T, typename Context>
void FloorDivideKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA FloorDivideKernel";
  dev_ctx.template Alloc<T>(out);

  doFloorDivTensor<T, Context>(dev_ctx, x, y, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(floor_divide,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::FloorDivideKernel,
                          int,
                          int64_t) {}