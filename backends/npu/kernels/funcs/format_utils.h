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

#ifndef __NATIVE_NPU_UTILS_FORMAT_HELPER__
#define __NATIVE_NPU_UTILS_FORMAT_HELPER__

#include "acl/acl.h"
#include "glog/logging.h"
#include "paddle/phi/extension.h"

namespace npu {

using FormatShape = std::vector<int64_t>;

// std::ostream& operator<<(std::ostream & out, FormatShape shape) {
//   int i = 0;
//   out << "[";
//   for(auto e : shape) {
//     if (i++ > 0)
//       out << ", ";
//     out << e;
//   }
//   out << "]";
//   return out;
// }

class FormatHelper {
public:
  static FormatShape GetStorageShape(const aclFormat storage_format, const FormatShape origin_dims);

private:
  static char* GetFormatName(const aclFormat& format);
  static std::string GetShapeString(const FormatShape& shape);

private:
  using shapeInfer = std::function<FormatShape(FormatShape dims)>;
  typedef struct FormatInfo_ {
    aclFormat format = ACL_FORMAT_ND;
    aclFormat baseFormat = ACL_FORMAT_ND;
    shapeInfer func = nullptr;
    char formatName[30] = {0};
    bool isPadded = false;
  } FormatInfo;
  static std::unordered_map<aclFormat, FormatInfo> info;
};

} // npu
#endif