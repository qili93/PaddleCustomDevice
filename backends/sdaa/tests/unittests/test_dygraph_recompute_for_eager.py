# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import random
import unittest

import numpy as np

import paddle
from paddle.distributed.fleet.utils import recompute

paddle.set_device("sdaa")


class Model(paddle.nn.Layer):
    def __init__(self, block_idx, input_size, is_last=False):
        super().__init__()
        block_name = "block_" + str(block_idx)
        self.block = paddle.nn.Sequential(
            (
                block_name + "_fc_0",
                paddle.nn.Linear(input_size, input_size, bias_attr=False),
            ),
            (block_name + "_dropout", paddle.nn.Dropout(p=0.5)),
            (block_name + "_relu_1", paddle.nn.ReLU()),
            (
                block_name + "_fc_1",
                paddle.nn.Linear(input_size, input_size, bias_attr=False),
            ),
            (block_name + "_relu_2", paddle.nn.ReLU()),
        )
        if is_last:
            self.block.add_sublayer(
                block_name + "_fc_2",
                paddle.nn.Linear(input_size, 1, bias_attr=False),
            )  # add sublayer
        else:
            self.block.add_sublayer(
                block_name + "_fc_2",
                paddle.nn.Linear(input_size, input_size, bias_attr=False),
            )  # add sublayer

    # add pos param for test kwargs of recompute.
    def forward(self, x, pos=None):
        if pos is None:
            return self.block(x)
        else:
            return self.block(x) + pos


def get_fc_block(block_idx, input_size, is_last=False):
    return Model(block_idx, input_size, is_last=False)


class Naive_fc_net(paddle.nn.Layer):
    def __init__(
        self,
        input_size=10,
        recompute_blocks=[1, 3],
        use_fleet_sq=False,
        segments=1,
        use_raw_recompute=False,
        recompute_kwargs={},
    ):
        super().__init__()
        self.recompute_blocks = recompute_blocks
        self.recompute_kwargs = recompute_kwargs
        self.use_fleet_sq = use_fleet_sq
        self.use_raw_recompute = use_raw_recompute
        self.segments = segments

        self.runfunc0 = get_fc_block(0, input_size, is_last=False)
        self.runfunc1 = get_fc_block(1, input_size, is_last=False)
        self.runfunc2 = get_fc_block(2, input_size, is_last=False)
        self.runfunc3 = get_fc_block(3, input_size, is_last=False)
        self.runfunc4 = get_fc_block(4, input_size, is_last=True)

        if self.use_fleet_sq and not use_raw_recompute:
            self.runfuncs = paddle.nn.Sequential(
                self.runfunc0,
                self.runfunc1,
                self.runfunc2,
                self.runfunc3,
                self.runfunc4,
            )

        self.layers = [
            self.runfunc0,
            self.runfunc1,
            self.runfunc2,
            self.runfunc3,
            self.runfunc4,
        ]

        # default segments = 2
        if use_raw_recompute:
            self.layers = [
                paddle.nn.Sequential(self.runfunc0, self.runfunc1),
                paddle.nn.Sequential(self.runfunc2, self.runfunc3, self.runfunc4),
            ]

    def forward(self, inputs):

        if self.use_fleet_sq and not self.use_raw_recompute:
            return paddle.incubate.distributed.fleet.recompute_sequential(
                {"segments": self.segments}, self.runfuncs, inputs
            )

        if self.use_raw_recompute:
            inputs = recompute(self.layers[0], inputs)
            return self.layers[1](inputs)

        for i in range(len(self.layers)):
            if i in self.recompute_blocks:
                inputs = recompute(self.layers[i], inputs, **self.recompute_kwargs)
            else:
                inputs = self.layers[i](inputs)

        return inputs


def run_model(
    recompute_block=[],
    recompute_kwargs={},
    use_fleet_sq=False,
    use_raw_recompute=False,
    segments=1,
    enable_autocast=False,
    pure_fp16=False,
):
    gen = paddle.seed(10)
    gen.manual_seed(10)
    np.random.seed(10)
    random.seed(10)

    batch_size, input_size = 1, 10
    model = Naive_fc_net(
        input_size,
        recompute_blocks=recompute_block,
        use_fleet_sq=use_fleet_sq,
        use_raw_recompute=use_raw_recompute,
        segments=segments,
        recompute_kwargs=recompute_kwargs,
    )

    if pure_fp16:
        model = paddle.amp.decorate(models=model, level="O2")

    loss_fn = paddle.nn.MSELoss(reduction="mean")
    optimizer = paddle.optimizer.Momentum(
        learning_rate=0.01, parameters=model.parameters()
    )

    if enable_autocast:
        scaler = paddle.amp.GradScaler()

    loss_ = []
    param_ = []
    grad_ = []
    for step in range(10):

        x_data = np.random.randn(batch_size, input_size).astype(np.float32)
        x = paddle.to_tensor(x_data)
        x.stop_gradient = False
        level = "O2" if pure_fp16 else "O1"
        with paddle.amp.auto_cast(True, level=level):
            y_pred = model(x)
            loss = y_pred.mean()
        if enable_autocast:
            scaler.scale(loss).backward()
            scaler.minimize(optimizer, loss)
        else:
            loss_.append(np.asarray(loss).tolist())
            loss.backward()
            optimizer.step()

        param_.append(np.asarray(model.parameters()[9]).tolist())
        grad_.append(np.asarray(model.parameters()[3]._grad_ivar()).tolist())

        optimizer.clear_grad()
    return loss_, param_, grad_


class TestRecompute(unittest.TestCase):
    def test_base_case(self, enable_autocast=False, pure_fp16=False):
        def check_identical(loss_ref, param_ref, grad_ref, loss, param, grad):
            self.assertEqual(loss_ref, loss)
            self.assertEqual(param_ref, param)
            self.assertEqual(grad_ref, grad)

        # without recompute
        loss_ref, param_ref, grad_ref = run_model(
            recompute_block=[],
            enable_autocast=enable_autocast,
            pure_fp16=pure_fp16,
        )

        # test for recompute
        # True: PyLayer of recompute
        # False: HooK of recompute
        for flag in [True, False]:
            # recompute second block
            loss, param, grad = run_model(
                recompute_block=[1],
                enable_autocast=enable_autocast,
                pure_fp16=pure_fp16,
                recompute_kwargs={"use_reentrant": flag},
            )
            check_identical(loss_ref, param_ref, grad_ref, loss, param, grad)

            # recompute fourth block
            loss, param, grad = run_model(
                recompute_block=[3],
                enable_autocast=enable_autocast,
                pure_fp16=pure_fp16,
                recompute_kwargs={"use_reentrant": flag},
            )
            check_identical(loss_ref, param_ref, grad_ref, loss, param, grad)

            # recompute second to fourth block
            loss, param, grad = run_model(
                recompute_block=[1, 2, 3],
                enable_autocast=enable_autocast,
                pure_fp16=pure_fp16,
                recompute_kwargs={"use_reentrant": flag},
            )
            check_identical(loss_ref, param_ref, grad_ref, loss, param, grad)

            # recompute second & fourth block
            loss, param, grad = run_model(
                recompute_block=[1, 3],
                enable_autocast=enable_autocast,
                pure_fp16=pure_fp16,
                recompute_kwargs={"use_reentrant": flag},
            )
            check_identical(loss_ref, param_ref, grad_ref, loss, param, grad)

            # recompute_sequential with segments=1 using fleet
            loss, param, grad = run_model(
                recompute_block=[],
                use_fleet_sq=True,
                enable_autocast=enable_autocast,
                pure_fp16=pure_fp16,
                recompute_kwargs={"use_reentrant": flag},
            )
            check_identical(loss_ref, param_ref, grad_ref, loss, param, grad)

        # with base recompute, and segments=2
        loss_ref, param_ref, grad_ref = run_model(
            recompute_block=[],
            enable_autocast=enable_autocast,
            use_raw_recompute=True,
            pure_fp16=pure_fp16,
        )

        # recompute using paddle.incubate.distributed.fleet.recompute_sequential, segments=2
        loss, param, grad = run_model(
            recompute_block=[],
            use_fleet_sq=True,
            segments=2,
            enable_autocast=enable_autocast,
            pure_fp16=pure_fp16,
        )
        check_identical(loss_ref, param_ref, grad_ref, loss, param, grad)

    def test_fc_net_with_dropout(self):
        self.test_base_case()

    def test_fc_net_without_restore_rng(self):
        for flag in [True, False]:
            loss_ref, param_ref, grad_ref = run_model(
                recompute_block=[2],
                recompute_kwargs={
                    "preserve_rng_state": False,
                    "use_reentrant": flag,
                },
                enable_autocast=True,
            )

    def test_fc_net_with_amp(self):
        self.test_base_case(enable_autocast=True)

    # pure_fp16 is not supported on sdaa.
    # def test_fc_net_with_fp16(self):
    #     self.test_base_case(enable_autocast=True, pure_fp16=True)

    def test_recompute_kwargs(self):
        pos = paddle.randn(shape=[10, 10], dtype="float32")
        pos.stop_gradient = False

        kwargs = {"pos": pos, "use_reentrant": True}
        with self.assertRaises(ValueError):
            loss_ref, param_ref, grad_ref = run_model(
                recompute_block=[2], recompute_kwargs=kwargs
            )

        kwargs = {"pos": pos, "use_reentrant": False}
        loss_ref, param_ref, grad_ref = run_model(
            recompute_block=[2], recompute_kwargs=kwargs
        )


if __name__ == "__main__":
    unittest.main()