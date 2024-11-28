import random
import numpy as np


def random_bool(probability):
    """
    Randomly return True or False based on a given probability.

    :param probability: Probability of returning True.
    :return: True or False.
    """
    return random.random() < probability


def sample_reverse_length(mean=4, std=1.0, min_len=2):
    """
    Sample reverse length from a truncated Gaussian distribution.

    :param mean: Mean length of the reverse sequence.
    :param std: Standard deviation of the reverse length.
    :param min_len: Minimum reverse length.
    :return: A sampled reverse length.
    """
    while True:
        length = int(round(np.random.normal(mean, std)))
        if min_len <= length:
            return length


class MixedTokenizer:
    def __init__(self, origin_tokenizer):
        """
        Initialize the MixedTokenizer with the original tokenizer.

        :param origin_tokenizer: The tokenizer to wrap and extend.
        """
        self.origin_tokenizer = origin_tokenizer
        self.resize_operated = False

        # Define special tokens for reverse encoding
        begin_of_reversed_box = "<BORB>"
        end_of_reversed_box = "<EORB>"

        # Check if special tokens are already in the vocabulary
        if begin_of_reversed_box not in self.origin_tokenizer.get_vocab():
            # Add new special tokens to the vocabulary
            new_tokens = [begin_of_reversed_box, end_of_reversed_box]
            num_added_tokens = self.origin_tokenizer.add_tokens(new_tokens)
            print(f"[LOG] Added {num_added_tokens} new tokens: {new_tokens}")
            self.resize_operated = True

        # Update vocabulary size
        self.vocab_size = len(self.origin_tokenizer)

        # Retrieve the IDs of the special tokens
        self.borb_token_id = self.origin_tokenizer.convert_tokens_to_ids(begin_of_reversed_box)
        self.eorb_token_id = self.origin_tokenizer.convert_tokens_to_ids(end_of_reversed_box)

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
        Delegate method and attribute access to the original tokenizer.

        :param name: The name of the attribute or method.
        :return: The corresponding method or attribute of the original tokenizer.
        """
        return getattr(self.origin_tokenizer, name)

    def encode(self, text, **kwargs):
        """
        Encode the input text using the original tokenizer.

        :param text: Input text to encode.
        :param kwargs: Additional arguments for the encode method.
        :return: Encoded token sequence.
        """
        return self.origin_tokenizer.encode(text, **kwargs)

    def decode(self, tokens):
        """
        Decode the token sequence, handling special tokens for reverse encoding.

        :param tokens: Input token sequence to decode.
        :return: Decoded text.
        """
        filtered_tokens = []
        is_in_box = False
        tmp = []

        for token in tokens:
            if token == self.borb_token_id:
                is_in_box = True
            elif token == self.eorb_token_id:
                is_in_box = False
                filtered_tokens.extend(tmp)
                tmp = []
            else:
                if is_in_box:
                    tmp.insert(0, token)  # Reverse the order of tokens inside the box
                else:
                    filtered_tokens.append(token)

        # Handle any unclosed reverse boxes
        if tmp:
            filtered_tokens.extend(tmp)

        decoded_text = self.origin_tokenizer.decode(filtered_tokens)
        return decoded_text

    def reverse_order_encode(self, text, add_special_tokens=False, **kwargs):
        """
        Encode the input text in reverse order and wrap it with special tokens.

        :param text: Input text to encode.
        :param add_special_tokens: Whether to add special tokens (e.g., <BOS>, <EOS>).
        :param kwargs: Additional arguments for the encode method.
        :return: Encoded token sequence with reverse order.
        """
        raw = self.origin_tokenizer.encode(text, add_special_tokens=False, **kwargs)
        encoded = []

        if add_special_tokens:
            encoded.append(self.origin_tokenizer.bos_token_id)

        encoded.append(self.borb_token_id)
        encoded.extend(list(reversed(raw)))
        encoded.append(self.eorb_token_id)

        if add_special_tokens:
            encoded.append(self.origin_tokenizer.eos_token_id)

        return encoded

    def mixed_order_encode(self, text, reverse_ratio=0.25, mean=4, std=1.0, min_len=2, add_special_tokens=False,
                           **kwargs):
        """
        Encode the input text with a mix of normal and reverse-ordered segments.

        :param text: Input text to encode.
        :param reverse_ratio: The probability of reversing a segment.
        :param mean: Mean length of reversed segments.
        :param std: Standard deviation of reversed segment lengths.
        :param min_len: Minimum length of reversed segments.
        :param add_special_tokens: Whether to add special tokens (e.g., <BOS>, <EOS>).
        :param kwargs: Additional arguments for the encode method.
        :return: Mixed-order encoded token sequence.
        """
        raw = self.origin_tokenizer.encode(text, add_special_tokens=False, **kwargs)
        encoded = []

        if add_special_tokens:
            encoded.append(self.origin_tokenizer.bos_token_id)

        idx = 0
        while idx < len(raw):
            r = raw[idx]
            # Decide whether to reverse a segment
            if idx <= len(raw) - min_len and random_bool(reverse_ratio / mean):
                length = sample_reverse_length(mean, std, min_len)
                encoded.append(self.borb_token_id)
                encoded.extend(reversed(raw[idx:idx + length]))
                encoded.append(self.eorb_token_id)
                idx += length
            else:
                encoded.append(r)
                idx += 1

        if add_special_tokens:
            encoded.append(self.origin_tokenizer.eos_token_id)

        return encoded


if __name__ == "__main__":
    from transformers import AutoTokenizer

    # Load the original tokenizer (e.g., for the Pythia model)
    model_name = "EleutherAI/pythia-70m"
    origin_tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize the MixedTokenizer
    mixed_tokenizer = MixedTokenizer(origin_tokenizer)

    # Test reverse order encoding and decoding
    test_text = "Hello world"

    # Reverse encoding
    reverse_encoded = mixed_tokenizer.reverse_order_encode(test_text)
    print(f"Reverse-encoded tokens: {reverse_encoded}")

    # Decoding
    reverse_decoded = mixed_tokenizer.decode(reverse_encoded)
    print(f"Decoded text: {reverse_decoded}")

    # Mixed-order encoding and decoding
    mixed_encoded = mixed_tokenizer.mixed_order_encode(test_text, reverse_ratio=0.5)
    print(f"Mixed-order encoded tokens: {mixed_encoded}")

    mixed_decoded = mixed_tokenizer.decode(mixed_encoded)
    print(f"Decoded text: {mixed_decoded}")
