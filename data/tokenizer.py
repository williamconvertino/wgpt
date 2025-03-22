import tiktoken

class Tokenizer:
    """
    A tokenizer class that uses tiktoken to encode and decode text.
    
    It utilizes a regular expression pattern for tokenization, incorporates special tokens,
    and supports encoding strings or lists of strings, as well as decoding sequences of token IDs.
    """
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

    def __init__(self):
        """
        Initializes the Tokenizer.
        
        Sets up the underlying tiktoken encoding with a custom regex pattern and special tokens.
        The special tokens include:
            - <|begin_of_text|>
            - <|end_of_text|>
            - <|pad|>
            
        It also calculates the total vocabulary size including the special tokens.
        """
        tokenizer_base = tiktoken.get_encoding("r50k_base")
        num_base_tokens = tokenizer_base.n_vocab

        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|pad|>"
        ]

        self.special_tokens = {
            token: i + num_base_tokens for i, token in enumerate(special_tokens)
        }

        self.eos_token_id = self.special_tokens["<|end_of_text|>"]
        self.bos_token_id = self.special_tokens["<|begin_of_text|>"]
        self.pad_token_id = self.special_tokens["<|pad|>"]

        self.tokenizer = tiktoken.Encoding(
            name="tokenizer",
            pat_str=self.pat_str,
            mergeable_ranks=tokenizer_base._mergeable_ranks,
            special_tokens=self.special_tokens
        )

        self.vocab_size = self.tokenizer.n_vocab + len(self.special_tokens)
    
    def __len__(self):
        """
        Returns the size of the vocabulary including special tokens.

        Returns:
            int: The total vocabulary size.
        """
        return self.vocab_size

    def _encode(self, text, eos=False, bos=False):
        """
        Encodes a single string into a list of token IDs with optional beginning and end tokens.
        
        Args:
            text (str): The input text string to encode.
            eos (bool, optional): Whether to append the end-of-sequence token. Defaults to False.
            bos (bool, optional): Whether to prepend the beginning-of-sequence token. Defaults to False.
        
        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        sequence = []
        
        if bos:
            sequence.append(self.special_tokens["<|begin_of_text|>"])
        
        sequence.extend(
            self.tokenizer.encode(
                text, 
                allowed_special=set(["<|begin_of_text|>", "<|end_of_text|>", "<|pad|>"])
            )
        )

        if eos:
            sequence.append(self.special_tokens["<|end_of_text|>"])
        
        return sequence

    def encode(self, text, eos=False, bos=False):
        """
        Encodes text into token IDs.
        
        If the input is a string, it returns a list of token IDs for that string.
        If the input is a list of strings, it returns a list of lists of token IDs.
        
        Args:
            text (str or list[str]): The text or list of texts to encode.
            eos (bool, optional): Whether to append the end-of-sequence token. Defaults to False.
            bos (bool, optional): Whether to prepend the beginning-of-sequence token. Defaults to False.
        
        Returns:
            list[int] or list[list[int]]: The encoded token IDs.
        
        Raises:
            ValueError: If the input type is not a string or list of strings.
        """
        if isinstance(text, str):
            return self._encode(text, eos, bos)
        elif isinstance(text, list):
            return [self._encode(t, eos, bos) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}, expected str or list")

    def decode(self, sequence):
        """
        Decodes a sequence of token IDs back into text.
        
        The input can be a list of integers or a list of lists of integers.
        
        Args:
            sequence (list[int] or list[list[int]]): The token IDs to decode.
        
        Returns:
            str or list[str]: The decoded text string, or list of decoded strings if a nested list is provided.
        
        Raises:
            ValueError: If the input is not a list of integers or a list of lists of integers.
        """
        if len(sequence) == 0:
            return ''
        if isinstance(sequence[0], list):
            return [self.tokenizer.decode(s) for s in sequence]
        elif isinstance(sequence[0], int):
            return self.tokenizer.decode(sequence)
        else:
            raise ValueError(f"Invalid input type: {type(sequence)}, expected list of lists or list of ints")
