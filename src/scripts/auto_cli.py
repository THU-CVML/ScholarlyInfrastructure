import os
import sys
import yaml
import shlex
import argparse
import subprocess
from typing import Dict, Any, List, Literal

class ConfigTranslator:
    # ... (此处与上文完全相同，保留 to_argparse 和 to_hydra 方法) ...
    @staticmethod
    def to_argparse(config: Dict[str, Any], prefix: str = "") -> List[str]:
        args = []
        for k, v in config.items():
            full_key = f"{prefix}.{k}" if prefix else k
            cli_flag = f"--{full_key}" if len(full_key) > 1 else f"-{full_key}"

            if isinstance(v, dict):
                args.extend(ConfigTranslator.to_argparse(v, prefix=full_key))
            elif isinstance(v, bool):
                if v: args.append(cli_flag)
            elif isinstance(v, list):
                args.append(cli_flag)
                args.extend([str(item) for item in v])
            elif v is not None:
                args.extend([cli_flag, str(v)])
        return args

    @staticmethod
    def to_hydra(config: Dict[str, Any], prefix: str = "") -> List[str]:
        args = []
        for k, v in config.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                args.extend(ConfigTranslator.to_hydra(v, prefix=full_key))
            elif isinstance(v, bool):
                args.append(f"{full_key}={str(v).lower()}")
            elif isinstance(v, list):
                list_str = "[" + ",".join(str(item) for item in v) + "]"
                args.append(f"{full_key}={list_str}")
            elif v is not None:
                args.append(f"{full_key}={v}")
        return args

class CLIWrapper:
    def __init__(
        self, 
        target_cmd: str, 
        yaml_filename: str, 
        configs_dir: str,
        output_dir: str,
        cli_style: Literal["argparse", "hydra"] = "argparse"
    ):
        self.target_cmd = target_cmd
        self.yaml_filename = yaml_filename
        self.configs_dir = configs_dir
        self.output_dir = output_dir
        self.cli_style = cli_style

    def _load_yaml(self) -> Dict[str, Any]:
        yaml_path = os.path.join(self.configs_dir, self.yaml_filename)
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"未找到配置文件: {yaml_path}")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    def _save_audit_config(self, config_dict: Dict[str, Any]):
        os.makedirs(self.output_dir, exist_ok=True)
        base_name, ext = os.path.splitext(self.yaml_filename)
        overlayed_name = f"{base_name}_overlayed{ext}"
        output_yaml_path = os.path.join(self.output_dir, overlayed_name)
        
        with open(output_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True, sort_keys=False)
        print(f"[AutoOptuna] 转译配置审计存入: {output_yaml_path}")

    def run(self, additional_args: List[str] = None):
        config_dict = self._load_yaml()
        self._save_audit_config(config_dict)
        
        if self.cli_style == "argparse":
            cli_args = ConfigTranslator.to_argparse(config_dict)
        elif self.cli_style == "hydra":
            cli_args = ConfigTranslator.to_hydra(config_dict)
        else:
            raise ValueError(f"不支持的 cli_style: {self.cli_style}")
            
        # 核心改动：使用 shlex 解析基础命令，支持复杂的启动器 (如 torchrun, swift sft)
        cmd = shlex.split(self.target_cmd) + cli_args
        if additional_args:
            cmd.extend(additional_args)
            
        print(f"[AutoOptuna] 执行命令: {' '.join(cmd)}")
        sys.exit(subprocess.run(cmd).returncode)


# ==========================================
# 核心：对外暴露的纯命令行入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoOptuna 通用配置转译启动器")
    
    # 我们自己工具需要的参数
    parser.add_argument("--target_cmd", type=str, default="python main.py",
    # required=True, 
                        help="底层目标命令 (例如: 'python train.py' 或 'torchrun --nproc_per_node=4 train.py')")
    parser.add_argument("--yaml_name", type=str, default="cli_params.yaml", 
                        help="配置目录中的 YAML 文件名 (例如: 'swift_params.yaml')")
    parser.add_argument("--cli_style", type=str, choices=["argparse", "hydra"], default="argparse", 
                        help="目标命令的参数风格")
    
    # AutoOptuna 框架传入的参数
    parser.add_argument("--configs_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # 允许透传其他未知参数到底层
    args, unknown_args = parser.parse_known_args()

    wrapper = CLIWrapper(
        target_cmd=args.target_cmd,
        yaml_filename=args.yaml_name,
        configs_dir=args.configs_dir,
        output_dir=args.output_dir,
        cli_style=args.cli_style
    )
    
    wrapper.run(additional_args=unknown_args)