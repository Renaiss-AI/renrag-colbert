"""
Advanced example of document indexing and searching with RenRAG ColBERT.

This example demonstrates:
- Working with a larger set of documents
- Creating, saving and reusing an index
- Advanced search patterns with different queries
- Performance considerations
"""

import os
import time
from renrag_colbert import ColbertIndexer, ColbertSearcher

# Sample academic paper abstracts
PAPER_ABSTRACTS = [
    {
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.",
        "authors": "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin",
        "year": 2017,
        "venue": "NeurIPS",
        "category": "NLP"
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks.",
        "authors": "Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova",
        "year": 2018,
        "venue": "NAACL",
        "category": "NLP"
    },
    {
        "title": "ImageNet Classification with Deep Convolutional Neural Networks",
        "abstract": "We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0% which is considerably better than the previous state-of-the-art. The neural network, which has 60 million parameters and 650,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and three fully-connected layers with a final 1000-way softmax.",
        "authors": "Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton",
        "year": 2012,
        "venue": "NeurIPS",
        "category": "Computer Vision"
    },
    {
        "title": "Deep Residual Learning for Image Recognition",
        "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth.",
        "authors": "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun",
        "year": 2016,
        "venue": "CVPR",
        "category": "Computer Vision"
    },
    {
        "title": "GPT-3: Language Models are Few-Shot Learners",
        "abstract": "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions – something which current NLP systems still largely struggle to do. Here we show that scaling language models greatly improves task-agnostic, few-shot performance.",
        "authors": "Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, et al.",
        "year": 2020,
        "venue": "NeurIPS",
        "category": "NLP"
    },
    {
        "title": "Generative Adversarial Networks",
        "abstract": "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere.",
        "authors": "Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio",
        "year": 2014,
        "venue": "NeurIPS",
        "category": "Generative Models"
    },
    {
        "title": "Proximal Policy Optimization Algorithms",
        "abstract": "We propose a new family of policy gradient methods for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a surrogate objective function using stochastic gradient ascent. Whereas standard policy gradient methods perform one gradient update per data sample, we propose a novel objective function that enables multiple epochs of minibatch updates. The new methods, which we call proximal policy optimization (PPO), have some of the benefits of trust region policy optimization (TRPO), but they are much simpler to implement, more general, and have better sample complexity.",
        "authors": "John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov",
        "year": 2017,
        "venue": "arXiv",
        "category": "Reinforcement Learning"
    },
    {
        "title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
        "abstract": "Deep neural nets with a large number of parameters are very powerful machine learning systems. However, overfitting is a serious problem in such networks. Large networks are also slow to use, making it difficult to deal with overfitting by combining the predictions of many different large neural nets at test time. Dropout is a technique for addressing this problem. The key idea is to randomly drop units (along with their connections) from the neural network during training. This prevents units from co-adapting too much.",
        "authors": "Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov",
        "year": 2014,
        "venue": "JMLR",
        "category": "Deep Learning"
    },
    {
        "title": "Adam: A Method for Stochastic Optimization",
        "abstract": "We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters. The method is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients.",
        "authors": "Diederik P. Kingma, Jimmy Ba",
        "year": 2015,
        "venue": "ICLR",
        "category": "Optimization"
    },
    {
        "title": "U-Net: Convolutional Networks for Biomedical Image Segmentation",
        "abstract": "There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method.",
        "authors": "Olaf Ronneberger, Philipp Fischer, Thomas Brox",
        "year": 2015,
        "venue": "MICCAI",
        "category": "Medical Imaging"
    }
]

def main():
    print("RenRAG ColBERT Advanced Search Example")
    print("-" * 70)
    
    # Initialize paths
    index_dir = os.path.join(os.getcwd(), "examples_data")
    index_name = "academic_papers"
    index_path = os.path.join(index_dir, index_name)
    
    # Ensure directory exists
    os.makedirs(index_dir, exist_ok=True)
    
    # Check if index already exists
    index_exists = os.path.exists(index_path) and os.path.isdir(index_path)
    
    # Initialize the indexer
    print("Initializing indexer...")
    indexer = ColbertIndexer(device="cpu")
    
    if not index_exists:
        print("\nCreating new academic papers index...")
        
        # Extract documents and metadata
        documents = [paper['abstract'] for paper in PAPER_ABSTRACTS]
        
        # Create file IDs based on paper titles (simplified for the example)
        file_ids = [f"{paper['title'].lower().replace(' ', '_')}.pdf" for paper in PAPER_ABSTRACTS]
        
        # Prepare metadata
        metadata = []
        for paper in PAPER_ABSTRACTS:
            metadata.append({
                "title": paper['title'],
                "authors": paper['authors'],
                "year": paper['year'],
                "venue": paper['venue'],
                "category": paper['category']
            })
        
        # Create doc IDs based on titles (simplified for the example)
        doc_ids = [f"paper-{i+1}" for i in range(len(documents))]
        
        # Time the indexing process
        start_time = time.time()
        
        # Create the index
        index_path = indexer.index(
            documents=documents,
            document_metadata=metadata,
            file_ids=file_ids,
            doc_ids=doc_ids,
            index_name=index_name,
            collection_dir=index_dir,
            overwrite=True
        )
        
        indexing_time = time.time() - start_time
        print(f"Index created at: {index_path}")
        print(f"Indexing time: {indexing_time:.2f} seconds for {len(documents)} documents")
    else:
        print(f"\nUsing existing index at: {index_path}")
    
    # Initialize the searcher
    print("\nInitializing searcher...")
    searcher = ColbertSearcher(index_path=index_path, device="cpu")
    
    # Demonstrate different types of queries
    queries = [
        "transformer models for natural language processing",
        "convolutional neural networks for image recognition",
        "generative adversarial networks",
        "optimization algorithms for deep learning",
        "methods to prevent overfitting in neural networks"
    ]
    
    # Search with each query
    print("\nPerforming searches with different queries:")
    for query in queries:
        print(f"\nQuery: \"{query}\"")
        
        # Time the search process
        start_time = time.time()
        results = searcher.search(query, k=3, threshold=0.1)
        search_time = time.time() - start_time
        
        print_academic_results(results)
        print(f"Search time: {search_time:.4f} seconds")
    
    # Demonstrate category filtering
    print("\n\nDemonstrating category filtering:")
    
    # Get all Computer Vision papers
    cv_file_ids = [paper['title'].lower().replace(' ', '_') + '.pdf' 
                  for paper in PAPER_ABSTRACTS 
                  if paper['category'] == 'Computer Vision']
    
    # Search only in Computer Vision papers
    print("\nSearching only in Computer Vision papers:")
    cv_results = searcher.search(
        "neural networks for images", 
        k=5, 
        filter_by_files=cv_file_ids
    )
    print_academic_results(cv_results)
    
    print("\nDone!")

def print_academic_results(results, show_metadata=True):
    print("-" * 70)
    if not results:
        print("No results found")
    else:
        for i, result in enumerate(results, 1):
            print(f"{i}. [Score: {result['score']:.4f}]")
            if show_metadata:
                print(f"   Title: {result['metadata'].get('title')}")
                print(f"   Authors: {result['metadata'].get('authors')}")
                print(f"   Year: {result['metadata'].get('year')}")
                print(f"   Venue: {result['metadata'].get('venue')}")
                print(f"   Category: {result['metadata'].get('category')}")
            print(f"   Abstract: {result['text'][:100]}...")
            print()
        print(f"Total results: {len(results)}")

if __name__ == "__main__":
    main()