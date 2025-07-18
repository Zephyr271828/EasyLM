import pprint
from functools import partial

import numpy as np
import mlxu

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
import optax
from transformers import (
    AutoTokenizer, GenerationConfig, FlaxLogitsProcessorList
)

from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.serving import LMServer
from EasyLM.jax_utils import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules, tree_apply,
    set_random_seed, get_float_dtype_by_name, make_shard_and_gather_fns,
    with_sharding_constraint, FlaxTemperatureLogitsWarper
)
from EasyLM.models.llama.llama_model import (
    LLaMAConfigurator, FlaxLLaMAForCausalLM
)

print("JAX devices:", jax.devices())

from jax_smi import initialise_tracking
initialise_tracking()

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    mesh_dim='1,-1,1',
    param_dtype='bf16',
    dtype='bf16',
    input_length=1024,
    seq_length=2048,
    top_k=50,
    top_p=1.0,
    do_sample=True,
    num_beams=1,
    add_bos_token=True,
    load_checkpoint='',
    tokenizer='openlm-research/open_llama_3b_v2',
    # llama=LLaMAConfigurator.get_default_config(),
    llama=LLaMAConfigurator.get_standard_llama_config('llama3_8b'),
    lm_server=LMServer.get_default_config(),
    jax_distributed=JaxDistributedConfig.get_default_config(),
)


def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    set_random_seed(FLAGS.seed)

    prefix_tokenizer = AutoTokenizer.from_pretrained(
        FLAGS.tokenizer, truncation_side='left', padding_side='left'
    )
    prefix_tokenizer.pad_token = prefix_tokenizer.eos_token
    tokenizer = AutoTokenizer.from_pretrained(
        FLAGS.tokenizer, truncation_side='right', padding_side='right'
    )
    tokenizer.pad_token = tokenizer.eos_token
    llama_config = LLaMAConfigurator.finalize_config(FLAGS.llama)
    print(llama_config)

    with jax.default_device(jax.devices("cpu")[0]):
        _, params = StreamingCheckpointer.load_trainstate_checkpoint(
            FLAGS.load_checkpoint, disallow_trainstate=True
        )

        hf_model = FlaxLLaMAForCausalLM(
            llama_config,
            input_shape=(1, FLAGS.seq_length),
            seed=FLAGS.seed,
            _do_init=False,
            dtype=get_float_dtype_by_name(FLAGS.dtype),
            param_dtype=get_float_dtype_by_name(FLAGS.param_dtype),
        )

    model_ps = match_partition_rules(
        LLaMAConfigurator.get_partition_rules(), params
    )
    shard_fns, _ = make_shard_and_gather_fns(
        model_ps, get_float_dtype_by_name(FLAGS.param_dtype)
    )

    # @partial(
    #     pjit,
    #     in_shardings=(model_ps, PS(), PS()),
    #     out_shardings=(PS(), PS(), PS())
    # )
    @partial(
        pjit,
        in_shardings=(
            model_ps,        # shard params along the fsdp axis
            PS(),            # rng is small — keep it fully replicated
            PS('dp', None),  # shard the batch’s leading axis over dp
        ),
        out_shardings=(
            PS('dp'),  # return loglikelihood sharded over dp
            PS(),            # is_greedy replicated
            PS(),            # new_rng replicated
        )
    )
    def forward_loglikelihood(params, rng, batch):
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        rng_generator = JaxRNG(rng)
        input_tokens = batch['input_tokens']
        output_tokens = batch['output_tokens']
        input_mask = batch['input_mask']
        output_mask = batch['output_mask']

        logits = hf_model.module.apply(
            params, input_tokens, attention_mask=input_mask,
            deterministic=True, rngs=rng_generator(LLaMAConfigurator.rng_keys()),
        ).logits
        loglikelihood = -optax.softmax_cross_entropy_with_integer_labels(
            logits, output_tokens
        )
        loglikelihood = jnp.sum(loglikelihood * output_mask, axis=-1)
        match_count = jnp.sum(
            (jnp.argmax(logits, axis=-1) == output_tokens) * output_mask,
            axis=-1
        )
        total = jnp.sum(output_mask, axis=-1)
        is_greedy = match_count == total
        return loglikelihood, is_greedy, rng_generator()


    @partial(
        pjit,
        in_shardings=(model_ps, PS(), PS(), PS()),
        out_shardings=(PS(), PS())
    )
    def forward_generate(params, rng, batch, temperature):
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        rng_generator = JaxRNG(rng)
        output = hf_model.generate(
            batch['input_tokens'],
            attention_mask=batch['attention_mask'],
            params=params['params'],
            prng_key=rng_generator(),
            logits_processor=FlaxLogitsProcessorList(
                [FlaxTemperatureLogitsWarper(temperature)]
            ),
            generation_config=GenerationConfig(
                max_new_tokens=FLAGS.seq_length - FLAGS.input_length,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=FLAGS.do_sample,
                num_beams=FLAGS.num_beams,
                top_k=FLAGS.top_k,
                top_p=FLAGS.top_p,
            )
        ).sequences[:, batch['input_tokens'].shape[1]:]
        return output, rng_generator()

    @partial(
        pjit,
        in_shardings=(model_ps, PS(), PS()),
        out_shardings=(PS(), PS())
    )
    def forward_greedy_generate(params, rng, batch):
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        rng_generator = JaxRNG(rng)
        output = hf_model.generate(
            batch['input_tokens'],
            attention_mask=batch['attention_mask'],
            params=params['params'],
            prng_key=rng_generator(),
            generation_config=GenerationConfig(
                max_new_tokens=FLAGS.seq_length - FLAGS.input_length,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1,
            )
        ).sequences[:, batch['input_tokens'].shape[1]:]
        return output, rng_generator()

    mesh = LLaMAConfigurator.get_jax_mesh(FLAGS.mesh_dim)
    # print(mesh)
    with mesh:
        params = tree_apply(shard_fns, params)
        sharded_rng = next_rng()

    class ModelServer(LMServer):

        @staticmethod
        def loglikelihood(prefix_text, text):
            nonlocal sharded_rng
            prefix = prefix_tokenizer(
                prefix_text,
                padding='max_length',
                truncation=True,
                max_length=FLAGS.input_length,
                return_tensors='np',
            )
            inputs = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=FLAGS.seq_length - FLAGS.input_length,
                return_tensors='np',
            )
            output_tokens = np.concatenate([prefix.input_ids, inputs.input_ids], axis=1)
            bos_tokens = np.full(
                (output_tokens.shape[0], 1), tokenizer.bos_token_id, dtype=np.int32
            )
            input_tokens = np.concatenate([bos_tokens, output_tokens[:, :-1]], axis=-1)
            input_mask = np.concatenate(
                [prefix.attention_mask, inputs.attention_mask], axis=1
            )
            if FLAGS.add_bos_token:
                bos_mask = np.ones_like(input_mask[:, :1])
            else:
                bos_mask = np.zeros_like(input_mask[:, :1])

            input_mask = np.concatenate([bos_mask, input_mask[:, :-1]], axis=1)
            output_mask = np.concatenate(
                [np.zeros_like(prefix.attention_mask), inputs.attention_mask], axis=1
            )
            batch = dict(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_mask=input_mask,
                output_mask=output_mask,
            )
            with mesh:
                loglikelihood, is_greedy, sharded_rng = forward_loglikelihood(
                    params, sharded_rng, batch
                )
                loglikelihood, is_greedy = jax.device_get((loglikelihood, is_greedy))
            return loglikelihood, is_greedy

        @staticmethod
        def loglikelihood_rolling(text):
            nonlocal sharded_rng
            inputs = tokenizer(
                text,
                padding='longest',
                truncation=False,
                max_length=np.iinfo(np.int32).max,
                return_tensors='np',
            )
            batch_size = inputs.input_ids.shape[0]
            output_tokens = inputs.input_ids
            attention_mask = inputs.attention_mask

            if output_tokens.shape[1] < FLAGS.seq_length:
                padding_length = FLAGS.seq_length - output_tokens.shape[1]
                pad_tokens = np.full(
                    (batch_size, padding_length), tokenizer.pad_token_id, dtype=np.int32
                )
                output_tokens = np.concatenate([output_tokens, pad_tokens], axis=-1)
                pad_mask = np.zeros(
                    (batch_size, padding_length), dtype=inputs.attention_mask.dtype
                )
                attention_mask = np.concatenate([attention_mask, pad_mask], axis=-1)

            bos_tokens = np.full(
                (batch_size, 1), tokenizer.bos_token_id, dtype=np.int32
            )
            input_tokens = np.concatenate([bos_tokens, output_tokens[:, :-1]], axis=-1)
            bos_mask = np.ones((batch_size, 1), dtype=inputs.attention_mask.dtype)
            total_seq_length = output_tokens.shape[1]

            total_loglikelihood = 0.0
            total_is_greedy = True
            # Sliding window
            for i in range(0, total_seq_length, FLAGS.seq_length):
                # Last window
                if i + FLAGS.seq_length > total_seq_length:
                    last_output_mask = np.copy(attention_mask[:, -FLAGS.seq_length:])
                    last_output_mask[:, :i - total_seq_length] = 0.0

                    batch = dict(
                        input_tokens=input_tokens[:, -FLAGS.seq_length:],
                        output_tokens=output_tokens[:, -FLAGS.seq_length:],
                        input_mask=attention_mask[:, -FLAGS.seq_length:],
                        output_mask=last_output_mask,
                    )

                # Normal window
                else:
                    batch = dict(
                        input_tokens=input_tokens[:, i:i + FLAGS.seq_length],
                        output_tokens=output_tokens[:, i:i + FLAGS.seq_length],
                        input_mask=attention_mask[:, i:i + FLAGS.seq_length],
                        output_mask=attention_mask[:, i:i + FLAGS.seq_length],
                    )

                with mesh:
                    loglikelihood, is_greedy, sharded_rng = forward_loglikelihood(
                        params, sharded_rng, batch
                    )
                    loglikelihood, is_greedy = jax.device_get((loglikelihood, is_greedy))

                total_loglikelihood += loglikelihood
                total_is_greedy = np.logical_and(is_greedy, total_is_greedy)

            return total_loglikelihood, total_is_greedy

        @staticmethod
        def generate(text, temperature):
            nonlocal sharded_rng
            inputs = prefix_tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=FLAGS.input_length,
                return_tensors='np',
            )
            input_tokens = inputs.input_ids
            input_mask = inputs.attention_mask
            if FLAGS.add_bos_token:
                input_tokens[:, 0] = tokenizer.bos_token_id
                input_mask[:, 0] = 1
            batch = dict(
                input_tokens=input_tokens,
                attention_mask=input_mask,
            )
            with mesh:
                output, sharded_rng = forward_generate(
                    params, sharded_rng, batch, temperature
                )
                output = jax.device_get(output)
            output_text = []
            for text in list(tokenizer.batch_decode(output)):
                if tokenizer.eos_token in text:
                    text = text.split(tokenizer.eos_token, maxsplit=1)[0]
                output_text.append(text)

            return output_text

        @staticmethod
        def greedy_until(prefix_text, until, max_length):
            nonlocal sharded_rng
            all_outputs = []
            for pf, ut in zip(prefix_text, until):
                if isinstance(ut, str):
                    ut = [ut]
                total_length = 0
                total_generated = ''

                while total_length < max_length:
                    pf_tokens = tokenizer(
                        pf,
                        padding=False,
                        truncation=False,
                        max_length=np.iinfo(np.int32).max,
                        return_tensors='np',
                    )
                    input_tokens = pf_tokens.input_ids
                    attention_mask = pf_tokens.attention_mask

                    if input_tokens.shape[1] < FLAGS.input_length:
                        extra = FLAGS.input_length - input_tokens.shape[1]
                        pad_tokens = np.full(
                            (1, extra), tokenizer.pad_token_id, dtype=np.int32
                        )
                        input_tokens = np.concatenate(
                            [pad_tokens, input_tokens], axis=1
                        )
                        pad_attention = np.zeros((1, extra), dtype=attention_mask.dtype)
                        attention_mask = np.concatenate(
                            [pad_attention, attention_mask], axis=1
                        )
                    elif input_tokens.shape[1] > FLAGS.input_length:
                        input_tokens = input_tokens[:, -FLAGS.input_length:]
                        attention_mask = attention_mask[:, -FLAGS.input_length:]

                    if FLAGS.add_bos_token:
                        input_tokens[:, 0] = tokenizer.bos_token_id
                        attention_mask[:, 0] = 1

                    batch = dict(input_tokens=input_tokens, attention_mask=attention_mask)

                    with mesh:
                        output, sharded_rng = forward_greedy_generate(
                            params, sharded_rng, batch
                        )
                        output = jax.device_get(output)

                    total_length += output.shape[1]
                    output_text = tokenizer.batch_decode(output)[0]
                    total_generated = total_generated + output_text
                    pf = pf + output_text

                    done = False
                    for s in ut:
                        if s in total_generated:
                            total_generated = total_generated.split(s, maxsplit=1)[0]
                            done = True
                    if done:
                        break

                all_outputs.append(total_generated)

            return all_outputs


    server = ModelServer(FLAGS.lm_server)
    server.run()


if __name__ == "__main__":
    mlxu.run(main)
