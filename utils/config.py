from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_config_args(parser: argparse.ArgumentParser, section: str) -> argparse.Namespace:
    parser.add_argument("--config", type=Path, help="YAML 配置文件路径")
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path)
    selected, _ = probe.parse_known_args()
    if selected.config is None:
        return parser.parse_args()

    import yaml

    config = yaml.safe_load(selected.config.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict) or section not in config:
        raise ValueError(f"配置 {selected.config} 缺少 {section} 部分")
    values = config[section]
    if not isinstance(values, dict):
        raise ValueError(f"配置 {selected.config} 中的 {section} 必须是映射")

    actions = {action.dest: action for action in parser._actions}
    # 显式命令行参数优先于 YAML 默认值。
    overrides = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("-")}
    defaults = {}
    for key, value in values.items():
        action = actions.get(key)
        if action is None:
            raise ValueError(f"配置项 {key!r} 不适用于 {section}")
        if any(option in overrides for option in action.option_strings):
            continue
        if action.type is not None and value is not None:
            convert = action.type
            value = [convert(item) for item in value] if isinstance(value, list) else convert(value)
        defaults[key] = value
    parser.set_defaults(**defaults)
    return parser.parse_args()
