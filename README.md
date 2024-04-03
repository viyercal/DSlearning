# DSlearning
A repo that stores some of the datascience learning projects I've embarked on, including building LLMs &amp; NNs from scratch

Quick notes on each project:

NanoGPT LLM:
  Implementation of Andrej Karpathy's NanoGPT complete with notes taken during the implementation process. Relatively unstructured, but it discusses the tradeoffs of various approaches that could be taken. The model compiles and runs, and produces Shakespearean style text (albeit in a primitive way, due to character-level tokenization and the code running on CPU vs GPU. **Note: the LLM is not currently GPT-2 Level, but if scaled up it could be (I don't have the compute resources).**
  While this architecture is decoder only, I have an acute understanding of the 'Attention is All You Need' Google Deepmind paper and how even modern day LLMs are implemented with MOE approaches, RAG, quantization, etc. at a conceptual or code-base level.

Neural Network from Scratch:
  Implementation of Samson Zhang's neural network from scratch Youtube video, using numpy as the foundational package as opposed to Pytorch or Tensorflow. Implemented to ground my understanding of the mathmatics behind neural networks, activation functions, etc. 
