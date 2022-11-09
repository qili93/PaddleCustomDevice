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

void OpPreparation::PrepareTensorWithFormat(phi::DenseTensor& tensor, aclFormat format) {
  // LOG(INFO) << "PrepareTensorWithFormat: tensor = " << npu::OpPreparation::DebugString(tensor);
  // LOG(INFO) << "PrepareTensorWithFormat: format = " << (int)format;
  if (tensor.storage_properties_initialized()) return;

  auto tensor_format = ConvertToNpuFormat(tensor.layout());
  FormatShape origin_shape = phi::vectorize<int64_t>(tensor.dims()); // [1, 8 ,5, 5]
  FormatShape storage_shape = FormatHelper::GetStorageShape(format, origin_shape); // [1, 1, 5, 5, 16]

  auto npu_properties = std::make_unique<phi::NPUStorageProperties>();
  npu_properties->origin_format = static_cast<int64_t>(tensor_format);
  npu_properties->storage_format = static_cast<int64_t>(format);
  npu_properties->origin_dims = tensor.dims();
  npu_properties->storage_dims = phi::make_ddim(storage_shape);
  tensor.set_storage_properties(std::move(npu_properties));

  // tensor.set_origin_format(static_cast<int64_t>(tensor_format));
  // tensor.set_storage_format(static_cast<int64_t>(format));

  // npu::FormatShape origin_shape = phi::vectorize<int64_t>(tensor.dims()); // [1, 8 ,5, 5]
  // npu::FormatShape storage_shape = npu::FormatHelper::GetStorageShape(format, origin_shape); // [1, 1, 5, 5, 16]

  tensor.ResizeAndAllocate(phi::make_ddim(storage_shape));

  // LOG(INFO) << "PrepareTensorWithFormat: tensor = " << npu::OpPreparation::DebugString(tensor);
}

// inline aclFormat GetStorageFormat(const phi::DenseTensor& tensor) {
//   if (tensor.storage_properties_initialized()) {
//     auto npu_properties = tensor.storage_properties<phi::NPUStorageProperties>();
//     return static_cast<aclFormat>(npu_properties.storage_format);
//   }
//   return ConvertToNpuFormat(tensor.layout());
// }

void OpPreparation::PrepareTensorWithTensor(const phi::DenseTensor& src, phi::DenseTensor& dst) {
  // aclFormat storage_format = GetStorageFormat(src);
  // PrepareTensorWithFormat(dst, storage_format);
  auto src_npu_properties = src.storage_properties<phi::NPUStorageProperties>();

  LOG(INFO) << "0 - PrepareTensorWithTensor: tensor = " << npu::OpPreparation::DebugString(dst);

  auto npu_properties = std::make_unique<phi::NPUStorageProperties>();
  npu_properties->origin_format = src_npu_properties.origin_format;
  npu_properties->storage_format = src_npu_properties.storage_format;
  npu_properties->origin_dims = src_npu_properties.origin_dims;
  npu_properties->storage_dims = src_npu_properties.storage_dims;
  dst.set_storage_properties(std::move(npu_properties));

  LOG(INFO) << "1 - PrepareTensorWithTensor: tensor = " << npu::OpPreparation::DebugString(dst);

  // core dump here????
  // dst.ResizeAndAllocate(npu_properties->storage_dims);
}

} // npu
