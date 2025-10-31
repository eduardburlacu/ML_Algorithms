import transformers
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors


model, tokenizer = load("Qwen/Qwen3-8B-MLX-4bit")
prompt = "Hello, please tell me what Qwen3 MLX series is for and how it differs from the vanilla Qwen3 model. Use <thinking> in your answer to reflect on your capabilities and limitations."

if tokenizer.chat_template is not None:
    messages = [
        {"role": "system", "content": "You are a helpful assistant that values honesty, factual information, be concise when the question does not require detailed information. If the user specifically mentions using or not using <thinking>, follow their instruction. If the thinking process results in the conclusion that you are not certain about the answer, say 'I don't know'."},
        {"role": "user", "content": prompt}
    
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

sampler = make_sampler(
    temp=0.75,
    top_k=30,
    top_p=0.9,
)

logits_processors = make_logits_processors(
    repetition_penalty=1.2, 
    repetition_context_size=32,
)

response = generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=True,
    max_tokens=4096,
    #sampler=sampler,
    #logits_processors=logits_processors
)

print(response)
