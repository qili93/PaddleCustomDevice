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

#include "kernels/funcs/npu_op_prepare.h"

namespace npu {

int64_t OpPreparation::PrepareTensorWithFormat(phi::DenseTensor& tensor, aclFormat format) {
  PADDLE_ENFORCE(tensor.valid(), phi::errors::InvalidArgument("The input tensor of PrepareTensorWithFormat must be valid."));

  VLOG(1) << "0 - PrepareTensorWithFormat with format: " << (int)(format) << ", tensor" << DebugString(tensor);

  if (tensor.storage_properties_initialized()) return 0;

  auto tensor_format = ConvertToNpuFormat(tensor.layout());
  auto npu_properties = std::make_unique<phi::NPUStorageProperties>();
  // Condition 1: origin format must be same with tensor layout
  npu_properties->origin_format = static_cast<int64_t>(tensor_format);
  // Condition 2: origin dims must be same with tensor dims
  npu_properties->origin_dims = tensor.dims();
  if (tensor_format == format) {
    npu_properties->storage_format = npu_properties->origin_format;
    npu_properties->storage_dims = npu_properties->origin_dims;
  } else {
    FormatShape origin_shape = phi::vectorize<int64_t>(tensor.dims()); // [1, 8 ,5, 5]
    FormatShape storage_shape = FormatHelper::GetStorageShape(format, origin_shape); // [1, 1, 5, 5, 16]
    npu_properties->storage_format = static_cast<int64_t>(format);
    npu_properties->storage_dims = phi::make_ddim(storage_shape);
  }
  // get requested_size before move unique ptr
  int64_t requested_size = phi::product(npu_properties->storage_dims);
  tensor.set_storage_properties(std::move(npu_properties));

  VLOG(1) << "1 - PrepareTensorWithFormat with format: " << (int)(format) << ", tensor: " << DebugString(tensor);

  // Condition 3: tensor capacity must align with actual storage dims

  return requested_size;
  // dev_ctx.template Alloc<T>(tensor, tensor.place(), requested_size);
}

} // npu