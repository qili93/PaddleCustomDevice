// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void NPUIdentityKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const int format,
                    phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  // auto src_format = ConvertToNpuFormat(x.layout());
  // auto origin_dims = phi::vectorize(x.dims());

  // auto dst_format = aclFormat(format);
  // auto storage_dims = phi::vectorize(out.dims());

  auto npu_properties = std::make_unique<phi::NPUStorageProperties>();
  npu_properties->origin_format = ConvertToNpuFormat(x.layout());
  npu_properties->storage_format = aclFormat(format);
  npu_properties->origin_dims = x.dims();
  npu_properties->storage_dims = out->dims();
  out->set_storage_properties(std::move(npu_properties));

  // npu::OpPreparation::PrepareTensorWithFormat(*out, aclFormat(format));
  // LOG(INFO) << "000000 - NPUIdentityKernel - Output out: " << npu::OpPreparation::DebugString(*out);
  NpuOpRunner runner_identity;
  runner_identity.SetType("Identity")
      .AddInput(x)
      .AddOutput(*out)
      .Run(stream);
}

}  // namespace custom_kernel


PD_REGISTER_PLUGIN_KERNEL(npu_identity,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::NPUIdentityKernel,
                          float,
                          double,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t,
                          bool,
                          phi::dtype::float16) {}
