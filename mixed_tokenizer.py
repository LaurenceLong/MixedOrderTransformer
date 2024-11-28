from typing import List, Optional, Union, Dict, Tuple

import numpy as np
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerBase
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING

from config import model_name


def random_bool(p: float) -> bool:
    return np.random.random() < p


def sample_reverse_length(mean: float, std: float, min_len: int) -> int:
    return max(min_len, int(np.random.normal(mean, std)))


class MixedTokenizer:
    # 添加这些类属性
    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {}  # 如果有特定的词表文件，在这里定义
    pretrained_vocab_files_map = {}  # 预训练词表文件映射
    max_model_input_sizes = {}  # 最大输入大小映射

    def __init__(self, origin_tokenizer, **kwargs):
        self.origin_tokenizer = origin_tokenizer

        # 直接从原始分词器获取必要的属性
        kwargs.setdefault('model_max_length', origin_tokenizer.model_max_length)
        kwargs.setdefault('padding_side', origin_tokenizer.padding_side)
        kwargs.setdefault('pad_token', origin_tokenizer.pad_token)

        # 复制原始分词器的基本属性
        for attr in ['bos_token', 'eos_token', 'unk_token', 'sep_token',
                     'pad_token', 'cls_token', 'mask_token',
                     'bos_token_id', 'eos_token_id', 'unk_token_id', 'sep_token_id',
                     'pad_token_id', 'cls_token_id', 'mask_token_id']:
            if hasattr(origin_tokenizer, attr):
                kwargs[attr] = getattr(origin_tokenizer, attr)

        self.resize_operated = False

        # 定义反转编码的特殊标记
        self.begin_of_reversed_box = "<BORB>"
        self.end_of_reversed_box = "<EORB>"

        # 添加特殊标记到词汇表
        special_tokens = [self.begin_of_reversed_box, self.end_of_reversed_box]
        if self.begin_of_reversed_box not in self.origin_tokenizer.get_vocab():
            num_added = self.origin_tokenizer.add_tokens(special_tokens)
            print(f"[LOG] Added {num_added} special tokens: {special_tokens}")
            self.resize_operated = True

        # Retrieve the IDs of the special tokens
        self.borb_token_id = self.origin_tokenizer.convert_tokens_to_ids(self.begin_of_reversed_box)
        self.eorb_token_id = self.origin_tokenizer.convert_tokens_to_ids(self.end_of_reversed_box)

        # Directly copy attributes from the original tokenizer
        self.bos_token_id = self.origin_tokenizer.bos_token_id
        self.eos_token_id = self.origin_tokenizer.eos_token_id
        self.unk_token_id = self.origin_tokenizer.unk_token_id
        self.sep_token_id = self.origin_tokenizer.sep_token_id
        self.pad_token_id = self.origin_tokenizer.pad_token_id
        self.pad_token_type_id = self.origin_tokenizer.pad_token_type_id
        self.cls_token_id = self.origin_tokenizer.cls_token_id
        self.mask_token_id = self.origin_tokenizer.mask_token_id
        self.additional_special_tokens_ids = self.origin_tokenizer.additional_special_tokens_ids

    def __getattr__(self, name):
        """
        代理未定义的属性和方法到原始分词器。
        """
        # 防止递归调用
        if name == 'origin_tokenizer':
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute 'origin_tokenizer'")

        if hasattr(self.origin_tokenizer, name):
            return getattr(self.origin_tokenizer, name)
        raise AttributeError(
            f"Neither '{self.__class__.__name__}' nor its original tokenizer "
            f"has attribute '{name}'"
        )

    @property
    def vocab_size(self) -> int:
        return len(self.origin_tokenizer)

    def encode(
            self,
            text: str,
            add_special_tokens: bool = True,
            **kwargs
    ) -> List[int]:
        """编码文本"""
        return self.origin_tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            **kwargs
        )

    def decode(
            self,
            token_ids: List[int],
            skip_special_tokens: bool = True,
            **kwargs
    ) -> str:
        """解码token序列，处理反转编码的特殊标记"""
        filtered_tokens = []
        is_in_box = False
        tmp = []

        for token in token_ids:
            if token == self.borb_token_id:
                is_in_box = True
            elif token == self.eorb_token_id:
                is_in_box = False
                filtered_tokens.extend(reversed(tmp))
                tmp = []
            else:
                if is_in_box:
                    tmp.append(token)
                else:
                    filtered_tokens.append(token)

        if tmp:  # 处理未闭合的反转框
            filtered_tokens.extend(reversed(tmp))

        return self.origin_tokenizer.decode(
            filtered_tokens,
            skip_special_tokens=skip_special_tokens,
            **kwargs
        )

    def reverse_order_encode(
            self,
            text: str,
            add_special_tokens: bool = False,
            padding: bool = False,
            truncation: bool = False,
            max_length: Optional[int] = None,
            return_tensors: Optional[str] = None,
            **kwargs
    ) -> Union[List[int], Dict]:
        """
        将输入文本反序编码并用特殊标记包装。

        Args:
            text: 输入文本
            add_special_tokens: 是否添加特殊标记（如BOS、EOS）
            padding: 是否填充
            truncation: 是否截断
            max_length: 最大长度
            return_tensors: 返回张量类型
            **kwargs: 传递给encode的额外参数

        Returns:
            反序编码后的token序列
        """
        # 基础编码
        raw = self.origin_tokenizer.encode(
            text,
            add_special_tokens=False,
            **kwargs
        )

        encoded = []

        # 添加BOS标记
        if add_special_tokens and self.bos_token_id is not None:
            encoded.append(self.bos_token_id)

        # 添加反转标记和反转序列
        encoded.append(self.borb_token_id)
        encoded.extend(reversed(raw))
        encoded.append(self.eorb_token_id)

        # 添加EOS标记
        if add_special_tokens and self.eos_token_id is not None:
            encoded.append(self.eos_token_id)

        # 处理padding和truncation
        if padding or truncation:
            encoded = self._pad_truncate(
                encoded,
                max_length=max_length,
                padding=padding,
                truncation=truncation
            )

        # 转换为张量
        if return_tensors:
            import torch
            encoded = torch.tensor(encoded)

        return encoded

    def mixed_order_encode(
            self,
            text: str,
            reverse_ratio: float = 0.25,
            mean: float = 4,
            std: float = 1.0,
            min_len: int = 2,
            add_special_tokens: bool = False,
            padding: bool = False,
            truncation: bool = False,
            max_length: Optional[int] = None,
            return_tensors: Optional[str] = None,
            **kwargs
    ) -> Union[List[int], Dict]:
        """
        混合正序和反序编码文本。

        Args:
            text: 输入文本
            reverse_ratio: 反转概率
            mean: 反转段落的平均长度
            std: 反转段落长度的标准差
            min_len: 最小反转长度
            add_special_tokens: 是否添加特殊标记
            padding: 是否填充
            truncation: 是否截断
            max_length: 最大长度
            return_tensors: 返回张量类型
            **kwargs: 额外参数

        Returns:
            编码后的token序列或字典
        """
        # 基础编码
        raw = self.origin_tokenizer.encode(
            text,
            add_special_tokens=False,
            **kwargs
        )

        encoded = []
        if add_special_tokens and self.bos_token_id is not None:
            encoded.append(self.bos_token_id)

        # 混合编码处理
        idx = 0
        while idx < len(raw):
            if (idx <= len(raw) - min_len and
                    random_bool(reverse_ratio / mean)):
                # 反转段落
                length = min(
                    sample_reverse_length(mean, std, min_len),
                    len(raw) - idx
                )
                encoded.append(self.borb_token_id)
                encoded.extend(reversed(raw[idx:idx + length]))
                encoded.append(self.eorb_token_id)
                idx += length
            else:
                encoded.append(raw[idx])
                idx += 1

        if add_special_tokens and self.eos_token_id is not None:
            encoded.append(self.eos_token_id)

        # 处理padding和truncation
        if padding or truncation:
            encoded = self._pad_truncate(
                encoded,
                max_length=max_length,
                padding=padding,
                truncation=truncation
            )

        # 转换为张量
        if return_tensors:
            import torch
            encoded = torch.tensor(encoded)

        return encoded

    def _pad_truncate(
            self,
            ids: List[int],
            max_length: Optional[int] = None,
            padding: bool = False,
            truncation: bool = False
    ) -> List[int]:
        """填充和截断处理"""
        if truncation and max_length and len(ids) > max_length:
            ids = ids[:max_length]

        if padding and max_length and len(ids) < max_length:
            pad_length = max_length - len(ids)
            if self.padding_side == "right":
                ids = ids + [self.pad_token_id] * pad_length
            else:
                ids = [self.pad_token_id] * pad_length + ids

        return ids


if __name__ == "__main__":

    # Load the original tokenizer (e.g., for the Pythia model)
    origin_tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize the MixedTokenizer
    mixed_tokenizer = MixedTokenizer(origin_tokenizer)

    # Test reverse order encoding and decoding
    test_text = "Hello world 1 + 1 = 2; 2 + 111 = 113; 1024 + 1024 = 2048"

    # Reverse encoding
    reverse_encoded = mixed_tokenizer.reverse_order_encode(test_text)
    print(f"Reverse-encoded tokens: {reverse_encoded}")

    # Decoding
    reverse_decoded = mixed_tokenizer.decode(reverse_encoded)
    print(f"Decoded text: {reverse_decoded}")

    for i in range(10):
        # Mixed-order encoding and decoding
        mixed_encoded = mixed_tokenizer.mixed_order_encode(test_text, reverse_ratio=0.25)
        print(f"Mixed-order encoded tokens: {mixed_encoded}")

        mixed_decoded = mixed_tokenizer.decode(mixed_encoded)
        print(f"Decoded text: {mixed_decoded}")

        assert test_text == mixed_decoded

        flat_decoded = mixed_tokenizer.origin_tokenizer.decode(mixed_encoded)
        print(f"Raw decode {flat_decoded}")


class MixedTokenizerConfig(PretrainedConfig):
    model_type = "mixed_tokenizer"


# 将配置和分词器添加到AUTO类映射中
CONFIG_MAPPING.register("mixed_tokenizer", MixedTokenizerConfig)
TOKENIZER_MAPPING.register(MixedTokenizerConfig, (MixedTokenizer, None))


# 修改加载代码
def load_tokenizer(output_dir):
    base_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    return MixedTokenizer(base_tokenizer)
