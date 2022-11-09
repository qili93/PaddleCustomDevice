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
void BatchNormKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& running_mean,
                     const phi::DenseTensor& running_var,
                     const phi::DenseTensor& scale,
                     const phi::DenseTensor& bias,
                     bool is_test,
                     float momentum,
                     float epsilon,
                     const std::string& data_layout_str,
                     bool use_global_stats,
                     bool trainable_stats,
                     phi::DenseTensor* y,
                     phi::DenseTensor* mean_out,
                     phi::DenseTensor* variance_out,
                     phi::DenseTensor* saved_mean,
                     phi::DenseTensor* saved_variance,
                     phi::DenseTensor* reserve_space) {
  if (x.storage_properties_initialized()) {
    auto npu_properties = x.storage_properties<phi::NPUStorageProperties>();
    phi::DenseTensorMeta y_meta = {x.dtype(), npu_properties.origin_dims, x.layout()};
    y->set_meta(y_meta);

    phi::DDim o_dims = phi::make_ddim({npu_properties.origin_dims[1]});
    phi::DenseTensorMeta o_meta = {x.dtype(), o_dims, x.layout()};
    mean_out->set_meta(o_meta);
    variance_out->set_meta(o_meta);
    saved_mean->set_meta(o_meta);
    saved_variance->set_meta(o_meta);
  }
  npu::OpPreparation::PrepareTensorWithFormat(*y, ACL_FORMAT_NC1HWC0);

  VLOG(1)<< "0 - BatchNormKernel - Input: x" << npu::OpPreparation::DebugString(x);
  VLOG(1)<< "0 - BatchNormKernel - Input: running_mean" << npu::OpPreparation::DebugString(running_mean);
  VLOG(1)<< "0 - BatchNormKernel - Input: running_var" << npu::OpPreparation::DebugString(running_var);
  VLOG(1)<< "0 - BatchNormKernel - Input: scale" << npu::OpPreparation::DebugString(scale);
  VLOG(1)<< "0 - BatchNormKernel - Input: bias" << npu::OpPreparation::DebugString(bias);
  VLOG(1)<< "0 - BatchNormKernel - Output: y" << npu::OpPreparation::DebugString(*y);
  VLOG(1)<< "0 - BatchNormKernel - Output: mean_out" << npu::OpPreparation::DebugString(*mean_out);
  VLOG(1)<< "0 - BatchNormKernel - Output: variance_out" << npu::OpPreparation::DebugString(*variance_out);
  VLOG(1)<< "0 - BatchNormKernel - Output: saved_mean" << npu::OpPreparation::DebugString(*saved_mean);
  VLOG(1)<< "0 - BatchNormKernel - Output: saved_variance" << npu::OpPreparation::DebugString(*saved_variance);

  VLOG(1)<< "--------------------------------------------------------------------------";

  bool test_mode = is_test && (!trainable_stats);
  bool training = !test_mode && !use_global_stats;

  phi::DataLayout data_layout = phi::StringToDataLayout(data_layout_str);

  const auto& x_dims = x.dims();
  // PADDLE_ENFORCE_EQ((x_dims.size() == 4UL || x_dims.size() == 3UL),
  //                   true,
  //                   phi::errors::InvalidArgument(
  //                       "The input tensor X's dimension must equal to 3 or 4. "
  //                       " But got X's shape = [%s], X's dimension = [%d].",
  //                       x_dims.to_str(),
  //                       x_dims.size()));

  dev_ctx.template Alloc<T>(y);

  phi::DenseTensor x_tensor(x), y_tensor(*y);
  // if (data_layout == phi::DataLayout::kNHWC) {
  //   phi::DenseTensorMeta x_meta = {x.dtype(), x.dims(), phi::DataLayout::kNHWC};
  //   phi::DenseTensorMeta y_meta = {y->dtype(), y->dims(), phi::DataLayout::kNHWC};
  //   x_tensor.set_meta(x_meta);
  //   y_tensor.set_meta(y_meta);
  // }

  auto stream = dev_ctx.stream();
  if (!training) {
    const auto& runner_infer =
        NpuOpRunner("BNInfer",
                    {x_tensor, scale, bias, running_mean, running_var},
                    {y_tensor},
                    {{"epsilon", epsilon}});
    runner_infer.Run(stream);
  } else {
    // dev_ctx.template Alloc<float>(y);
    dev_ctx.template Alloc<float>(mean_out);
    dev_ctx.template Alloc<float>(variance_out);
    dev_ctx.template Alloc<float>(saved_mean);
    dev_ctx.template Alloc<float>(saved_variance);

    // phi::DenseTensor sum, square_sum;
    // sum.Resize(running_mean.dims());
    // square_sum.Resize(running_mean.dims());
    // dev_ctx.template Alloc<float>(&sum);
    // dev_ctx.template Alloc<float>(&square_sum);
    // BNTrainingReduce ONLY support rank = 4
    if (x.dims().size() == 3) {
      auto x_shape_vec = phi::vectorize(x.dims());
      if (data_layout == phi::DataLayout::kNCHW) {
        x_shape_vec.push_back(1);  // expand NCL -> NCL1
      } else {
        x_shape_vec.insert(x_shape_vec.begin() + 2, 1);  // expand NLC -> NL1C
      }
      auto x_new_shape = phi::make_ddim(x_shape_vec);
      x_tensor.Resize(x_new_shape);
      x_tensor.Resize(x_new_shape);
    }
    // const auto& runner_reduce = NpuOpRunner("BNTrainingReduce",
    //                                         {x_tensor},
    //                                         {sum, square_sum},
    //                                         {{"epsilon", epsilon}});
    // runner_reduce.Run(stream);

    // const auto& runner_update = NpuOpRunner(
    //     "BNTrainingUpdate",
    //     {x_tensor, sum, square_sum, scale, bias, running_mean, running_var},
    //     {y_tensor, *mean_out, *variance_out, *saved_mean, *saved_variance},
    //     {{"factor", momentum}, {"epsilon", epsilon}});
    // runner_update.Run(stream);

    // prepare tmp output of sum and square_sum in storage format
    // npu::FormatShape origin_shape = phi::vectorize<int64_t>(running_mean.dims()); // [1, 8 ,5, 5]
    // npu::FormatShape storage_shape = npu::FormatHelper::GetStorageShape(ACL_FORMAT_NC1HWC0, origin_shape); // [1, 1, 5, 5, 16]

    phi::DenseTensorMeta meta = {x.dtype(), mean_out->dims(), x.layout()};
    phi::DenseTensor sum, square_sum;
    sum.set_meta(meta);
    square_sum.set_meta(meta);
    dev_ctx.template Alloc<T>(&sum);
    dev_ctx.template Alloc<T>(&square_sum);

    // VLOG(1)<< "!!!!!!!! running_mean.dims()" << running_mean.dims();
    // VLOG(1)<< "!!!!!!!! sum.dims()" << sum.dims();
    // VLOG(1)<< "!!!!!!!! square_sum.dims()" << square_sum.dims();

    // npu::OpPreparation::PrepareTensorWithFormat(x_tensor, ACL_FORMAT_NCHW); // conv2d 的输出，本来格式就是对的
    npu::OpPreparation::PrepareTensorWithFormat(sum, ACL_FORMAT_NC1HWC0);
    npu::OpPreparation::PrepareTensorWithFormat(square_sum, ACL_FORMAT_NC1HWC0);

    VLOG(1)<< "0 - BNTrainingReduce - Input: x_tensor" << npu::OpPreparation::DebugString(x_tensor);
    VLOG(1)<< "0 - BNTrainingReduce - Output: sum" << npu::OpPreparation::DebugString(sum);
    VLOG(1)<< "0 - BNTrainingReduce - Output: square_sum" << npu::OpPreparation::DebugString(square_sum);

    // sum.Resize(phi::make_ddim(storage_shape));
    // square_sum.Resize(phi::make_ddim(storage_shape));
    // dev_ctx.template Alloc<T>(&sum);
    // dev_ctx.template Alloc<T>(&square_sum);

    NpuOpRunner runner_reduce;
    runner_reduce.SetType("BNTrainingReduce")
        .AddInput(x_tensor)
        .AddOutput(sum)
        .AddOutput(square_sum)
        .AddAttrs({{"epsilon", epsilon}})
        .Run(stream);

    VLOG(1)<< "1 - BNTrainingReduce - Input: x_tensor" << npu::OpPreparation::DebugString(x_tensor);
    VLOG(1)<< "1 - BNTrainingReduce - Output: sum" << npu::OpPreparation::DebugString(sum);
    VLOG(1)<< "1 - BNTrainingReduce - Output: square_sum" << npu::OpPreparation::DebugString(square_sum);

    VLOG(1)<< "--------------------------------------------------------------------------";

    phi::DenseTensor scale_tensor(scale), bias_tensor(bias), running_mean_tensor(running_mean), running_var_tensor(running_var);
    VLOG(1)<< "0 - BN2D - Input: scale" << npu::OpPreparation::DebugString(scale_tensor);
    VLOG(1)<< "0 - BN2D - Input: bias" << npu::OpPreparation::DebugString(bias_tensor);
    VLOG(1)<< "0 - BN2D - Input: running_mean" << npu::OpPreparation::DebugString(running_mean_tensor);
    VLOG(1)<< "0 - BN2D - Input: running_var" << npu::OpPreparation::DebugString(running_var_tensor);
    npu::OpPreparation::PrepareTensorWithFormat(scale_tensor, ACL_FORMAT_NC1HWC0);
    npu::OpPreparation::PrepareTensorWithFormat(bias_tensor, ACL_FORMAT_NC1HWC0);
    npu::OpPreparation::PrepareTensorWithFormat(running_mean_tensor, ACL_FORMAT_NC1HWC0);
    npu::OpPreparation::PrepareTensorWithFormat(running_var_tensor, ACL_FORMAT_NC1HWC0);
    VLOG(1)<< "1 - BN2D - Input: scale" << npu::OpPreparation::DebugString(scale_tensor);
    VLOG(1)<< "1 - BN2D - Input: bias" << npu::OpPreparation::DebugString(bias_tensor);
    VLOG(1)<< "1 - BN2D - Input: running_mean" << npu::OpPreparation::DebugString(running_mean_tensor);
    VLOG(1)<< "1 - BN2D - Input: running_var" << npu::OpPreparation::DebugString(running_var_tensor);

    phi::DenseTensor mean_out_tensor(*mean_out), var_out_tensor(*variance_out), saved_mean_tensor(*saved_mean), saved_var_tensor(*saved_variance);
    VLOG(1)<< "0 - BN2D - Output: mean_out" << npu::OpPreparation::DebugString(mean_out_tensor);
    VLOG(1)<< "0 - BN2D - Output: variance_out" << npu::OpPreparation::DebugString(var_out_tensor);
    VLOG(1)<< "0 - BN2D - Output: saved_mean" << npu::OpPreparation::DebugString(saved_mean_tensor);
    VLOG(1)<< "0 - BN2D - Output: saved_variance" << npu::OpPreparation::DebugString(saved_var_tensor);
    npu::OpPreparation::PrepareTensorWithFormat(mean_out_tensor, ACL_FORMAT_NC1HWC0);
    npu::OpPreparation::PrepareTensorWithFormat(var_out_tensor, ACL_FORMAT_NC1HWC0);
    npu::OpPreparation::PrepareTensorWithFormat(saved_mean_tensor, ACL_FORMAT_NC1HWC0);
    npu::OpPreparation::PrepareTensorWithFormat(saved_var_tensor, ACL_FORMAT_NC1HWC0);
    VLOG(1)<< "1 - BN2D - Output: mean_out" << npu::OpPreparation::DebugString(mean_out_tensor);
    VLOG(1)<< "1 - BN2D - Output: variance_out" << npu::OpPreparation::DebugString(var_out_tensor);
    VLOG(1)<< "1 - BN2D - Output: saved_mean" << npu::OpPreparation::DebugString(saved_mean_tensor);
    VLOG(1)<< "1 - BN2D - Output: saved_variance" << npu::OpPreparation::DebugString(saved_var_tensor);


    VLOG(1)<< "0 - BN2D - Input: x" << npu::OpPreparation::DebugString(x_tensor);
    npu::OpPreparation::PrepareTensorWithFormat(x_tensor, ACL_FORMAT_NC1HWC0);
    VLOG(1)<< "1 - BN2D - Input: x" << npu::OpPreparation::DebugString(x_tensor);

    VLOG(1)<< "0 - BN2D - Output: y" << npu::OpPreparation::DebugString(y_tensor);
    // npu::OpPreparation::PrepareTensorWithTensor(x_tensor, y_tensor);
    npu::OpPreparation::PrepareTensorWithFormat(y_tensor, ACL_FORMAT_NC1HWC0);
    VLOG(1)<< "1 - BN2D - Output: y" << npu::OpPreparation::DebugString(y_tensor);



  //   // transform all weights to ACL_FORMAT_NC1HWC0
  //   phi::DenseTensor scale_tensor, bias_tensor, running_mean_tensor, running_var_tensor;
  //   scale_tensor.Resize(phi::make_ddim(storage_shape));
  //   bias_tensor.Resize(phi::make_ddim(storage_shape));
  //   running_mean_tensor.Resize(phi::make_ddim(storage_shape));
  //   running_var_tensor.Resize(phi::make_ddim(storage_shape));
  //   dev_ctx.template Alloc<float>(&scale_tensor);
  //   dev_ctx.template Alloc<float>(&bias_tensor);
  //   dev_ctx.template Alloc<float>(&running_mean_tensor);
  //   dev_ctx.template Alloc<float>(&running_var_tensor);
  //   NpuOpRunner runner_identity_scale;
  //   runner_identity_scale.SetType("Identity")
  //       .AddInput(scale)
  //       .AddOutput(scale_tensor, ACL_FORMAT_NCHW, origin_shape, ACL_FORMAT_NC1HWC0, storage_shape)
  //       .Run(stream);
  //   NpuOpRunner runner_identity_bias;
  //   runner_identity_bias.SetType("Identity")
  //       .AddInput(bias)
  //       .AddOutput(bias_tensor, ACL_FORMAT_NCHW, origin_shape, ACL_FORMAT_NC1HWC0, storage_shape)
  //       .Run(stream);
  //   NpuOpRunner runner_identity_mean;
  //   runner_identity_mean.SetType("Identity")
  //       .AddInput(running_mean)
  //       .AddOutput(running_mean_tensor, ACL_FORMAT_NCHW, origin_shape, ACL_FORMAT_NC1HWC0, storage_shape)
  //       .Run(stream);
  //   NpuOpRunner runner_identity_var;
  //   runner_identity_var.SetType("Identity")
  //       .AddInput(running_var)
  //       .AddOutput(running_var_tensor, ACL_FORMAT_NCHW, origin_shape, ACL_FORMAT_NC1HWC0, storage_shape)
  //       .Run(stream);

  //   // prepare all outputs in ACL_FORMAT_NC1HWC0 format
  //   phi::DenseTensor mean_out_tensor, var_out_tensor, saved_mean_tensor, saved_var_tensor;
  //   mean_out_tensor.Resize(phi::make_ddim(storage_shape));
  //   var_out_tensor.Resize(phi::make_ddim(storage_shape));
  //   saved_mean_tensor.Resize(phi::make_ddim(storage_shape));
  //   saved_var_tensor.Resize(phi::make_ddim(storage_shape));
  //   dev_ctx.template Alloc<T>(&mean_out_tensor);
  //   dev_ctx.template Alloc<T>(&var_out_tensor);
  //   dev_ctx.template Alloc<T>(&saved_mean_tensor);
  //   dev_ctx.template Alloc<T>(&saved_var_tensor);

  // // prepare temp output of storage format
  //   npu::FormatShape y_origin_shape = phi::vectorize<int64_t>(y_tensor.dims());
  //   npu::FormatShape y_storage_shape = npu::FormatHelper::GetStorageShape(ACL_FORMAT_NC1HWC0, y_origin_shape);
  //   phi::DenseTensor y_tensor_storage;
  //   y_tensor_storage.Resize(phi::make_ddim(y_storage_shape));
  //   dev_ctx.template Alloc<T>(&y_tensor_storage);


    NpuOpRunner runner_update;
    runner_update.SetType("BNTrainingUpdate")
        .AddInput(x_tensor)
        .AddInput(sum)
        .AddInput(square_sum)
        .AddInput(scale_tensor)
        .AddInput(bias_tensor)
        .AddInput(running_mean_tensor)
        .AddInput(running_var_tensor)
        .AddOutput(y_tensor)
        .AddOutput(mean_out_tensor)
        .AddOutput(var_out_tensor)
        .AddOutput(saved_mean_tensor)
        .AddOutput(saved_var_tensor)
        .AddAttrs({{"epsilon", static_cast<float>(epsilon)}})
        .AddAttrs({{"factor", static_cast<float>(momentum)}})
        .Run(stream);

  //   // transform output back to origin format
  //   NpuOpRunner runner_identity_y;
  //   runner_identity_y.SetType("Identity")
  //       .AddInput(y_tensor_storage, ACL_FORMAT_NCHW, y_origin_shape, ACL_FORMAT_NC1HWC0, y_storage_shape)
  //       .AddOutput(y_tensor)
  //       .Run(stream);
  //   // transform weight back to origin format
  //   NpuOpRunner runner_identity_mean_out;
  //   runner_identity_mean_out.SetType("Identity")
  //       .AddInput(mean_out_tensor, ACL_FORMAT_NCHW, y_origin_shape, ACL_FORMAT_NC1HWC0, y_storage_shape)
  //       .AddOutput(*mean_out)
  //       .Run(stream);
  //   NpuOpRunner runner_identity_var_out;
  //   runner_identity_var_out.SetType("Identity")
  //       .AddInput(var_out_tensor, ACL_FORMAT_NCHW, y_origin_shape, ACL_FORMAT_NC1HWC0, y_storage_shape)
  //       .AddOutput(*variance_out)
  //       .Run(stream);
  //   NpuOpRunner runner_identity_saved_mean;
  //   runner_identity_saved_mean.SetType("Identity")
  //       .AddInput(saved_mean_tensor, ACL_FORMAT_NCHW, y_origin_shape, ACL_FORMAT_NC1HWC0, y_storage_shape)
  //       .AddOutput(*saved_mean)
  //       .Run(stream);
  //   NpuOpRunner runner_identity_saved_var;
  //   runner_identity_saved_var.SetType("Identity")
  //       .AddInput(saved_var_tensor, ACL_FORMAT_NCHW, y_origin_shape, ACL_FORMAT_NC1HWC0, y_storage_shape)
  //       .AddOutput(*saved_variance)
  //       .Run(stream);
  }
}

template <typename T, typename Context>
void BatchNormGradKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const phi::DenseTensor& scale,
    const phi::DenseTensor& bias,
    const paddle::optional<phi::DenseTensor>& mean,
    const paddle::optional<phi::DenseTensor>& variance,
    const phi::DenseTensor& saved_mean,
    const phi::DenseTensor& saved_inv_variance,
    const paddle::optional<phi::DenseTensor>& reserve_space,
    const phi::DenseTensor& d_y,
    float momentum,
    float epsilon,
    const std::string& data_layout_str,
    bool is_test,
    bool use_global_stats,
    bool trainable_statistics,
    phi::DenseTensor* d_x,
    phi::DenseTensor* d_scale,
    phi::DenseTensor* d_bias) {




  phi::DataLayout data_layout = phi::StringToDataLayout(data_layout_str);

  use_global_stats = is_test || use_global_stats;

  phi::DenseTensor x_tensor(x), dy_tensor(d_y);
  if (data_layout == phi::DataLayout::kNHWC) {
    phi::DenseTensorMeta x_meta = {x.dtype(), x.dims(), phi::DataLayout::kNHWC};
    phi::DenseTensorMeta dy_meta = {
        d_y.dtype(), d_y.dims(), phi::DataLayout::kNHWC};
    x_tensor.set_meta(x_meta);
    dy_tensor.set_meta(dy_meta);
  }

  phi::DenseTensor scale_grad_tmp, bias_grad_tmp;
  scale_grad_tmp.Resize(scale.dims());
  bias_grad_tmp.Resize(bias.dims());
  dev_ctx.template Alloc<float>(&scale_grad_tmp);
  dev_ctx.template Alloc<float>(&bias_grad_tmp);

  if (d_scale == nullptr) {
    d_scale = &scale_grad_tmp;
  }
  if (d_bias == nullptr) {
    d_bias = &bias_grad_tmp;
  }

  auto stream = dev_ctx.stream();
  if (d_scale && d_bias) {
    dev_ctx.template Alloc<float>(d_scale);
    dev_ctx.template Alloc<float>(d_bias);

    if (use_global_stats) {
      const auto* running_mean = mean.get_ptr();
      const auto* running_variance = variance.get_ptr();
      const auto& runner_update =
          NpuOpRunner("BNTrainingUpdateGrad",
                      {dy_tensor, x_tensor, *running_mean, *running_variance},
                      {*d_scale, *d_bias},
                      {{"epsilon", epsilon}});
      runner_update.Run(stream);
    } else {
      const auto& runner_update =
          NpuOpRunner("BNTrainingUpdateGrad",
                      {dy_tensor, x_tensor, saved_mean, saved_inv_variance},
                      {*d_scale, *d_bias},
                      {{"epsilon", epsilon}});
      runner_update.Run(stream);
    }
  }
  if (d_x) {
    dev_ctx.template Alloc<T>(d_x);
    phi::DenseTensor dx_tensor(*d_x);
    if (data_layout == phi::DataLayout::kNHWC) {
      phi::DenseTensorMeta dx_meta = {
          d_x->dtype(), d_x->dims(), phi::DataLayout::kNHWC};
      dx_tensor.set_meta(dx_meta);
    }
    if (use_global_stats) {
      if (x.dims().size() == 3) {
        // BNInferGrad only support x rank = 4,
        auto x_shape_vec = phi::vectorize(d_x->dims());
        if (data_layout == phi::DataLayout::kNCHW) {
          x_shape_vec.push_back(1);  // expand NCL -> NCL1
        } else {
          x_shape_vec.insert(x_shape_vec.begin() + 2, 1);  // expand NLC -> NL1C
        }
        auto x_new_shape = phi::make_ddim(x_shape_vec);
        dx_tensor.Resize(x_new_shape);
        dy_tensor.Resize(x_new_shape);
      }
      const auto* running_variance = variance.get_ptr();
      const auto& runner_infer =
          NpuOpRunner("BNInferGrad",
                      {dy_tensor, scale, *running_variance},
                      {dx_tensor},
                      {{"epsilon", epsilon}});
      runner_infer.Run(stream);
    } else {
      const auto& runner_reduce = NpuOpRunner("BNTrainingReduceGrad",
                                              {dy_tensor,
                                               x_tensor,
                                               *d_scale,
                                               *d_bias,
                                               scale,
                                               saved_mean,
                                               saved_inv_variance},
                                              {dx_tensor},
                                              {{"epsilon", epsilon}});
      runner_reduce.Run(stream);
    }
  }
}

template <typename T, typename Context>
void BatchNormInferKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& mean,
                          const phi::DenseTensor& variance,
                          const phi::DenseTensor& scale,
                          const phi::DenseTensor& bias,
                          float momentum,
                          float epsilon,
                          const std::string& data_layout_str,
                          phi::DenseTensor* y,
                          phi::DenseTensor* mean_out,
                          phi::DenseTensor* variance_out) {
  phi::DataLayout data_layout = StringToDataLayout(data_layout_str);

  const auto& x_dims = x.dims();
  PADDLE_ENFORCE_EQ((x_dims.size() == 4UL || x_dims.size() == 3UL),
                    true,
                    phi::errors::InvalidArgument(
                        "The input tensor X's dimension must equal to 3 or 4. "
                        " But got X's shape = [%s], X's dimension = [%d].",
                        x_dims.to_str(),
                        x_dims.size()));

  dev_ctx.template Alloc<T>(y);

  phi::DenseTensor x_tensor(x);
  phi::DenseTensor y_tensor(*y);
  if (data_layout == phi::DataLayout::kNHWC) {
    phi::DenseTensorMeta x_meta = {x.dtype(), x.dims(), phi::DataLayout::kNHWC};
    phi::DenseTensorMeta y_meta = {
        y->dtype(), y->dims(), phi::DataLayout::kNHWC};
    x_tensor.set_meta(x_meta);
    y_tensor.set_meta(y_meta);
  }

  auto stream = dev_ctx.stream();
  const auto& runner_infer =
      NpuOpRunner("BNInfer",
                  {x_tensor, scale, bias, mean, variance},
                  {y_tensor},
                  {{"epsilon", epsilon}});
  runner_infer.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(batch_norm,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::BatchNormKernel,
                          phi::dtype::float16,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(batch_norm_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::BatchNormGradKernel,
                          phi::dtype::float16,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(batch_norm_infer,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::BatchNormInferKernel,
                          phi::dtype::float16,
                          float,
                          double) {}
