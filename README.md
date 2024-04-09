# DSlearning
A repo that stores some of the datascience learning projects I've embarked on, including building LLMs &amp; NNs from scratch. All in Python unless otherwise noted.

Quick notes on each project:

NanoGPT LLM:
  Implementation of Andrej Karpathy's NanoGPT complete with notes taken during the implementation process. Relatively unstructured, but it discusses the tradeoffs of various approaches that could be taken. The model compiles and runs, and produces Shakespearean style text (albeit in a fairly primitive way, due to character-level tokenization and the code running on CPU rather than a GPU. If you download the file, adjustable parameters that reflect a tradeoff between quality, performance, and speed include n_embd, n_head, context_window, and batch_size, among others. **Note: the LLM is not currently GPT-2 Level, but it could be with a few adjustments if scaled up (I don't have the compute resources).**
  While this architecture is decoder only, I have an acute understanding of the 'Attention is All You Need' Google Deepmind paper and how even modern day LLMs are implemented with MOE approaches, RAG, quantization, etc. at a conceptual or code-base level.

Neural Network from Scratch:
  Implementation of Samson Zhang's neural network from scratch Youtube video, using numpy as the foundational package as opposed to Pytorch or Tensorflow. Implemented to ground my understanding of the mathmatics behind neural networks, activation functions, etc. 

Image Classifier Cifar-10:
  Still in progress, features a custom convolutional neural network with a conv layer built from scratch (implementation from https://www.youtube.com/watch?v=Lakz2MoHy6o&ab_channel=TheIndependentCode). Implemented to learn the underlying mathmatics and design choices within a conv net, and used to bolster my understanding of computer vision basics.

Superintelligence & AI Agents: AI Existential Risk Implications:
  My 2023 paper on the existential risk implications presented by AI agents and the potential for superintelligent AI systems.


Other projects that I've built include:\
  Football betting ATS (against the spread) model leveraging neural networks, random forests & decision trees, XGBoost en route to accuracy surpassing 538's open source model\
  Pacman adversarial agent gameplay strategies\
  2D world generation & basic navigation/gameplay (Java)\
  Google N-gram Viewer functionality (Java)\
  Data structures projects related to graphs, trees, and linkedlists (Java)\
  Plants vs Zombies style game
  U.S. Semiconductor Export Restrictions to China: Viewed from the Lens of AI-risk Analysis (2024)

  The last six projects (after the football betting model) are school projects that are restricted in visibility due to some of the projects' elements being re-used each semester, however code and results can be provided upon request. 
  
