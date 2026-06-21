#!/usr/bin/env python3
"""
DataSculptor — Generate synthetic training data for fine-tuning AI agents.
Zero dependencies. Supports Q&A, classification, summarization, and custom formats.
"""

import json, random, argparse, os
from typing import List, Dict

TEMPLATES = {
    "qa": [
        {"question": "What is {topic}?", "answer": "{topic} is a field of {domain} that focuses on {focus}."},
        {"question": "How does {topic} work?", "answer": "{topic} works by {method}, which involves {steps}."},
        {"question": "Why is {topic} important?", "answer": "{topic} is important because {reason}. It enables {benefit}."},
        {"question": "What are the benefits of {topic}?", "answer": "The main benefits include {benefit1}, {benefit2}, and {benefit3}."},
        {"question": "Can you explain {topic} in simple terms?", "answer": "Think of {topic} as {analogy}. It helps you {outcome}."},
    ],
    "classification": [
        {"text": "The {product} is great for {use_case}.", "label": "positive"},
        {"text": "I don't like how {product} handles {feature}.", "label": "negative"},
        {"text": "{product} costs ${price} and does {function}.", "label": "neutral"},
        {"text": "Absolutely love {product}! Best {category} tool ever.", "label": "positive"},
        {"text": "Waste of money. {product} broke after {time}.", "label": "negative"},
    ],
    "summarization": [
        {"text": "{paragraph}", "summary": "This text discusses {topic} and its impact on {domain}."},
    ],
}

WORD_BANK = {
    "topic": ["machine learning", "neural networks", "transformers", "reinforcement learning", "computer vision", "NLP", "data science", "deep learning", "AI agents", "LLMs"],
    "domain": ["artificial intelligence", "computer science", "data engineering", "software development", "research"],
    "focus": ["pattern recognition", "automated reasoning", "statistical modeling", "optimization algorithms", "knowledge representation"],
    "method": ["training on large datasets", "iterative optimization", "gradient-based learning", "attention mechanisms", "self-supervised learning"],
    "steps": ["data collection, model training, and evaluation", "feature extraction, pattern matching, and prediction", "encoding, processing, and decoding information"],
    "reason": ["it automates complex tasks", "it finds patterns humans miss", "it scales to massive datasets", "it enables new capabilities"],
    "benefit": ["faster decision-making", "improved accuracy", "cost reduction", "new insights"],
    "benefit1": ["increased efficiency", "reduced costs", "better accuracy", "faster processing"],
    "benefit2": ["improved scalability", "enhanced reliability", "greater flexibility", "deeper insights"],
    "benefit3": ["automated workflows", "real-time analysis", "predictive capabilities", "personalized experiences"],
    "analogy": ["a smart assistant that learns from experience", "having a team of experts working 24/7", "a GPS that finds the best path through data"],
    "outcome": ["make better decisions", "save time and money", "discover hidden patterns", "automate repetitive work"],
    "product": ["AgentPro", "ModelForge", "DataFlow", "NeuralKit", "PromptGen"],
    "use_case": ["data analysis", "code generation", "content creation", "customer support", "research"],
    "feature": ["the interface", "response time", "accuracy", "pricing", "documentation"],
    "function": ["text generation", "image analysis", "data processing", "code review"],
    "price": ["10", "25", "50", "100", "500"],
    "category": ["AI", "developer", "productivity", "creative"],
    "time": ["2 weeks", "a month", "3 days", "6 hours"],
    "paragraph": [
        "Artificial intelligence has transformed how we process information. Modern systems can analyze vast datasets and extract meaningful patterns that were previously invisible to human analysts.",
        "The rise of large language models has created new possibilities for natural language understanding. These systems can now generate coherent text, answer complex questions, and even write code.",
        "Machine learning pipelines have become essential infrastructure for modern businesses. From recommendation systems to fraud detection, ML powers critical decision-making processes.",
    ],
}

def generate_dataset(template_type: str, count: int, output_file: str = None) -> List[Dict]:
    """Generate a synthetic dataset."""
    data = []
    templates = TEMPLATES.get(template_type, TEMPLATES["qa"])
    
    for i in range(count):
        template = random.choice(templates)
        entry = {}
        for key, value in template.items():
            if isinstance(value, str):
                filled = value
                # Replace placeholders
                for placeholder in WORD_BANK:
                    if "{" + placeholder + "}" in filled:
                        replacement = random.choice(WORD_BANK[placeholder])
                        filled = filled.replace("{" + placeholder + "}", replacement)
                entry[key] = filled
            elif isinstance(value, list):
                entry[key] = [random.choice(value)]
            else:
                entry[key] = value
        data.append(entry)
    
    if output_file:
        ext = os.path.splitext(output_file)[1]
        with open(output_file, "w") as f:
            if ext == ".jsonl":
                for item in data:
                    f.write(json.dumps(item) + "\n")
            else:
                json.dump(data, f, indent=2)
    
    return data

def main():
    parser = argparse.ArgumentParser(description="DataSculptor — Synthetic Training Data Generator")
    parser.add_argument("--type", "-t", choices=["qa", "classification", "summarization", "all"],
                       default="qa", help="Dataset type")
    parser.add_argument("--count", "-n", type=int, default=100, help="Number of examples")
    parser.add_argument("--output", "-o", help="Output file (.json or .jsonl)")
    
    args = parser.parse_args()
    
    print(f"\n🎨 DataSculptor — Generating {args.count} {args.type} examples\n")
    
    if args.type == "all":
        for dtype in ["qa", "classification", "summarization"]:
            count = args.count // 3
            data = generate_dataset(dtype, count)
            print(f"  {dtype}: {len(data)} examples generated")
            if args.output:
                base, ext = os.path.splitext(args.output)
                generate_dataset(dtype, count, f"{base}_{dtype}{ext}")
    else:
        data = generate_dataset(args.type, args.count, args.output)
        print(f"  Generated {len(data)} {args.type} examples")
        
        if not args.output:
            # Print sample
            print(f"\n  Sample:")
            for item in data[:3]:
                print(f"  {json.dumps(item)[:100]}...")
    
    if args.output:
        print(f"\n📄 Saved to {args.output}")
    print(f"\n💡 Tip: Use --type all to generate all formats, or specify --count for more examples.")

if __name__ == "__main__":
    main()
