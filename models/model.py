# Copyright (c) 2024 Stepfun AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Contains the classes necessary for doing PPO (offline, one-step) with language model.
This code is largely from the TRL library, with some modifications to ensure stability.
"""

import torch.nn as nn
from transformers import AutoModelForCausalLM

class PreTrainedModelWrapper(nn.Module):
    r"""
    A wrapper class around a (`transformers.PreTrainedModel`) to be compatible with the
    (`~transformers.PreTrained`) class in order to keep some attributes and methods of the
    (`~transformers.PreTrainedModel`) class.

    Attributes:
        pretrained_model: (`transformers.PreTrainedModel`)
            The model to be wrapped.
        transformers_parent_class: (`transformers.PreTrainedModel`)
            The parent class of the model to be wrapped.
    """
    transformers_parent_class = None
    def __init__(self, pretrained_model=None):
        super().__init__()
        self.pretrained_model = pretrained_model

    @classmethod
    def from_pretrained(cls, pretrained_model_path, *args, **kwargs):
        r"""
        Instantiates a new model from a pretrained model from `transformers`. The
        pretrained model is loaded using the `from_pretrained` method of the
        `transformers.PreTrainedModel` class. The arguments that are specific to the
        `transformers.PreTrainedModel` class are passed along this method and filtered
        out from the `kwargs` argument.

        Args:
            pretrained_model_path (`str`):
                The path to the pretrained model.
            **args (`tuple`, *optional*):
                Additional arguments passed along to the underlying model's
                `from_pretrained` method. 
            **kwargs (`dict`, *optional*):
                Additional keyword arguments passed along to the underlying model's
                `from_pretrained` method. 
        """

        # First, load the pre-trained model using the parent-class
        # either `AutoModelForCausalLM` or `AutoModelForSeq2SeqLM`
        if isinstance(pretrained_model_path, str):
            pretrained_model = cls.transformers_parent_class.from_pretrained(
                pretrained_model_path, *args, **kwargs
            )
        else:
            raise ValueError(
                "pretrained_model_name_or_path should be a string, "
                f"but is {type(pretrained_model_path)}"
            )
        # call the __init__ method to create an instance
        model = cls(pretrained_model)
        return model

class ScalarHead(nn.Module):
    r"""
    The ValueHead class implements a head for autoregressive that returns a scalar for each output token.
    The weights of the value head need to be in FP32.
    """

    def __init__(self, config, bias=True):
        super().__init__()
        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        else:
            hidden_size = config.hidden_size
        # use a linear head to output a scalar
        self.summary =  nn.Linear(hidden_size, 1, bias=bias)    


    def forward(self, hidden_states):
        output = self.summary(hidden_states)
        return output


class AutoModelForCausalLMWithScalarHead(PreTrainedModelWrapper):
    r"""
    An autoregressive model with a scalar head.
    This is used for reward model and critic model to output a scalar for reward/value

    Class attributes:
        - **transformers_parent_class** (`transformers.PreTrainedModel`) -- The parent class of the wrapped model. This
            should be set to `transformers.AutoModelForCausalLM` for this class.
        - **lm_head_namings** (`tuple`) -- A tuple of strings that are used to identify the language model head of the
            wrapped model. This is set to `("lm_head", "embed_out")` for this class but can be changed for other models
            in the future
    """
    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]
    def __init__(self, pretrained_model):
        r"""
        Initializes the model.

        Args:
            pretrained_model (`transformers.PreTrainedModel`):
                The model to wrap. It should be a causal language model such as GPT2.
                or any model mapped inside the `AutoModelForCausalLM` class.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class.
        """
        super().__init__(pretrained_model)
        if not any(hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        self.scalar_head = ScalarHead(self.pretrained_model.config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        **kwargs,
    ):
        r"""
        Applies a forward pass to the wrapped model and returns the values of the scalar head.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.

        Returns:
            scalar_values (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                scalar values given by causal LM with a scalar head
        """
        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = base_model_output.hidden_states[-1]

        scalar_value = self.scalar_head(last_hidden_state).squeeze(-1)
        return scalar_value
    
class AutoModelForCausalLMWithScalarHeadODIN(PreTrainedModelWrapper):
    r"""
    An autoregressive model with two scalar heads.
    This is used for reward model to output two scalars for rewards(one is quality, another is length)

    Class attributes:
        - **transformers_parent_class** (`transformers.PreTrainedModel`) -- The parent class of the wrapped model. This
            should be set to `transformers.AutoModelForCausalLM` for this class.
        - **lm_head_namings** (`tuple`) -- A tuple of strings that are used to identify the language model head of the
            wrapped model. This is set to `("lm_head", "embed_out")` for this class but can be changed for other models
            in the future
    """
    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]
    def __init__(self, pretrained_model):
        r"""
        Initializes the model.

        Args:
            pretrained_model (`transformers.PreTrainedModel`):
                The model to wrap. It should be a causal language model such as GPT2.
                or any model mapped inside the `AutoModelForCausalLM` class.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class.
        """
        super().__init__(pretrained_model)
        if not any(hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        self.scalar_head_quality = ScalarHead(self.pretrained_model.config, bias=False)
        self.scalar_head_length = ScalarHead(self.pretrained_model.config, bias=False)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        **kwargs,
    ):
        r"""
        Applies a forward pass to the wrapped model and returns the values of the scalar head.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.

        Returns:
            scalar_values (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                scalar values given by causal LM with a scalar head
        """
        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = base_model_output.hidden_states[-1]
        scalar_value_quality = self.scalar_head_quality(last_hidden_state).squeeze(-1)
        scalar_value_length = self.scalar_head_length(last_hidden_state).squeeze(-1)

        return scalar_value_quality, scalar_value_length