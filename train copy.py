"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

# 启用faulthandler，用于在程序崩溃时生成详细的堆栈跟踪，便于调试
import faulthandler
faulthandler.enable()

# 忽略所有警告信息，避免在训练过程中输出过多无关信息
import warnings
warnings.filterwarnings('ignore')

# 导入os模块，用于操作系统相关操作，如环境变量设置和路径处理
import os
# 设置环境变量，解决Intel MKL库在多线程环境下的重复加载问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 导入sys模块，用于系统路径操作
import sys
# 将上级目录添加到Python搜索路径中，以便导入本地模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# 导入json模块，用于配置的JSON序列化
import json
# 导入argparse模块，用于命令行参数解析
import argparse

# 从engine模块导入相关组件
from engine.logger_module import get_logger  # 日志记录器
from engine.extre_module.torch_utils import check_cuda  # CUDA可用性检查
from engine.misc import dist_utils  # 分布式训练工具
from engine.core import YAMLConfig, yaml_utils  # YAML配置管理
from engine.solver import TASKS  # 任务字典，包含不同训练任务的solver类

# 定义ANSI颜色码，用于终端输出高亮显示
RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
# 获取当前模块的日志记录器
logger = get_logger(__name__)
# 调试模式开关，默认为False
debug = False

# 如果启用调试模式，自定义PyTorch张量的字符串表示，显示形状信息
if debug:
    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

def main(args) -> None:
    """主函数，负责训练或验证的整体流程"""
    # 设置分布式训练环境，包括打印方法、进程ID和随机种子
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    # 检查CUDA是否可用
    check_cuda()

    # 断言：不能同时启用tuning和resume，只能选择从头训练、恢复或调优之一
    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    # 解析命令行中的更新参数，生成更新字典
    update_dict = yaml_utils.parse_cli(args.update)
    # 将其他非空参数（除update外）合并到更新字典中
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    # 创建YAML配置对象，加载配置文件并应用更新
    cfg = YAMLConfig(args.config, **update_dict)

    # 如果是恢复或调优模式，禁用HGNetv2的预训练权重
    if args.resume or args.tuning:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    # 将配置对象序列化为JSON字符串，并彩色打印
    cfg_str = json.dumps(cfg.__dict__, indent=4, ensure_ascii=False)
    print(GREEN + cfg_str + RESET)

    # 根据配置中的任务类型，从TASKS字典获取对应的solver类并实例化
    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    # 根据test_only参数决定执行训练还是验证
    if args.test_only:
        # 如果提供了模型路径，使用ONNX/Engine模型进行验证
        if args.path:
            solver.val_onnx_engine(args.mode)
        else:
            # 如果启用了WDA评估，在指定数据根目录下进行验证
            if args.eval_wda:
                data_root = '/home/wyq/wyq/DEIM-DEIM/data/gwhd_2021/test_domain'
                solver.val_wda(data_root)
            else:
                # 默认进行标准验证
                solver.val()
    else:
        # 执行训练
        solver.fit(cfg_str)

    # 清理分布式训练环境
    dist_utils.cleanup()

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()

    # 优先级0的参数（核心参数）
    parser.add_argument('-c', '--config', type=str, required=True, help='配置文件路径，必须提供')
    parser.add_argument('-r', '--resume', type=str, help='从检查点恢复训练')
    parser.add_argument('-t', '--tuning', type=str, help='从检查点调优训练')
    parser.add_argument('-d', '--device', type=str, help='指定设备，如GPU或CPU')
    parser.add_argument('--seed', type=int, help='随机种子，用于实验可复现性', default=42)
    parser.add_argument('--use-amp', action='store_true', help='启用自动混合精度训练')
    parser.add_argument('--output-dir', type=str, help='输出目录，用于保存模型和日志')
    parser.add_argument('--summary-dir', type=str, help='TensorBoard摘要目录')
    parser.add_argument('--test-only', action='store_true', default=False, help='仅执行测试/验证，不训练')

    # 测试相关参数
    parser.add_argument('-p', '--path', type=str, help='ONNX/Engine模型路径，仅在test-only模式下使用')
    parser.add_argument('-m', '--mode', type=str, default='det', choices=['det', 'mask'], help='Engine模型模式：det（检测）或mask（分割），仅在test-only模式下使用')

    # 优先级1的参数（配置更新）
    parser.add_argument('-u', '--update', nargs='+', help='更新YAML配置的参数列表')

    # 环境相关参数
    parser.add_argument('--print-method', type=str, default='builtin', help='打印方法')
    parser.add_argument('--print-rank', type=int, default=0, help='打印进程ID')
    parser.add_argument('--local-rank', type=int, help='本地进程ID，用于分布式训练')

    # 自定义参数：添加WDA评估标志
    parser.add_argument('--eval-wda', action='store_true', default=False, help='启用WDA（Weighted Domain Adaptation）评估')

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数
    main(args)