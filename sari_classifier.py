import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import open_clip
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class SariProductionClassifier:
    def __init__(self):
        """Initialize the production classifier"""
        print("🤖 Loading CLIP model...")
        
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Default configuration from your experiments
        self.prompt_text = "A red sari with evenly distributed floral and geometric motifs"
        self.blend_weight = 0.8  # Image-heavy approach (from seed 42)
        self.threshold = 0.4     # Acceptance threshold
        
        # Pre-extract prompt embedding
        self.prompt_embedding = self._extract_prompt_embedding()
        
        # Placeholder for trained classifier (we'll use similarity-based for now)
        self.classifier = None
        self.scaler = None
        
        print(f"✅ Sari Classifier ready on {self.device}")
    
    def _extract_prompt_embedding(self):
        """Extract CLIP embedding for the prompt text"""
        with torch.no_grad():
            text_tokens = self.tokenizer([self.prompt_text]).to(self.device)
            prompt_embedding = self.model.encode_text(text_tokens)
            return prompt_embedding.cpu().numpy().flatten()
    
    def _extract_image_embeddings(self, images):
        """Extract CLIP embeddings for a batch of images"""
        embeddings = []
        batch_size = 16  # Smaller batch for production
        
        print(f"🎨 Extracting features from {len(images)} images...")
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_tensors = []
                
                for img in batch_images:
                    try:
                        if isinstance(img, str):  # If it's a file path
                            img = Image.open(img).convert('RGB')
                        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
                        batch_tensors.append(tensor)
                    except Exception as e:
                        print(f"❌ Error processing image: {e}")
                        # Add zero tensor as fallback
                        batch_tensors.append(torch.zeros(1, 3, 224, 224).to(self.device))
                
                if batch_tensors:
                    batch_tensor = torch.cat(batch_tensors)
                    batch_embeddings = self.model.encode_image(batch_tensor)
                    embeddings.extend(batch_embeddings.cpu().numpy())
        
        return np.array(embeddings)
    
    def _calculate_similarity_scores(self, image_embeddings):
        """Calculate cosine similarity with prompt"""
        similarities = cosine_similarity(image_embeddings, self.prompt_embedding.reshape(1, -1))
        return similarities.flatten()
    
    def _calculate_confidence(self, similarity_scores, final_scores):
        """Calculate prediction confidence"""
        # Confidence based on distance from threshold
        confidence = np.abs(final_scores - self.threshold)
        # Normalize to 0-1 range
        confidence = np.clip(confidence * 2, 0, 1)
        return confidence
    
    def classify_images(self, image_inputs, image_names=None):
        """
        Classify multiple images
        
        Args:
            image_inputs: List of PIL Images or file paths
            image_names: List of filenames (optional)
        
        Returns:
            pandas DataFrame with results
        """
        if not image_names:
            image_names = [f"image_{i+1}.jpg" for i in range(len(image_inputs))]
        
        print(f"🔄 Processing {len(image_inputs)} images...")
        
        # Extract image embeddings
        image_embeddings = self._extract_image_embeddings(image_inputs)
        
        # Calculate similarity scores with prompt
        similarity_scores = self._calculate_similarity_scores(image_embeddings)
        
        # For production, use similarity-based scoring (you can add trained classifier later)
        # Normalize similarity scores from [-1, 1] to [0, 1]
        normalized_scores = (similarity_scores + 1) / 2
        
        # Apply threshold for predictions
        predictions = (normalized_scores >= self.threshold).astype(int)
        
        # Calculate confidence
        confidence_scores = self._calculate_confidence(similarity_scores, normalized_scores)
        
        # Create results dataframe
        results = pd.DataFrame({
            'filename': image_names,
            'similarity_score': similarity_scores,
            'final_score': normalized_scores,
            'prediction': ['Accept' if p == 1 else 'Reject' for p in predictions],
            'status': ['✅ Accept' if p == 1 else '❌ Reject' for p in predictions],
            'confidence': confidence_scores
        })
        
        # Sort by final score (best first)
        results = results.sort_values('final_score', ascending=False).reset_index(drop=True)
        
        accepted_count = sum(predictions)
        rejected_count = len(predictions) - accepted_count
        
        print(f"✅ Classification complete!")
        print(f"📊 Results: {accepted_count} Accepted, {rejected_count} Rejected")
        
        return results
    
    def update_settings(self, blend_weight=None, threshold=None, prompt_text=None):
        """Update classifier settings"""
        if blend_weight is not None:
            self.blend_weight = blend_weight
            print(f"🎛️ Blend weight updated to: {blend_weight}")
        
        if threshold is not None:
            self.threshold = threshold
            print(f"🎯 Threshold updated to: {threshold}")
        
        if prompt_text is not None and prompt_text != self.prompt_text:
            self.prompt_text = prompt_text
            print(f"📝 Prompt updated to: '{prompt_text}'")
            print("🔄 Re-extracting prompt embedding...")
            self.prompt_embedding = self._extract_prompt_embedding()
        
    def get_summary_stats(self, results):
        """Get summary statistics from results"""
        total = len(results)
        accepted = len(results[results['prediction'] == 'Accept'])
        rejected = total - accepted
        avg_score = results['final_score'].mean()
        
        return {
            'total_images': total,
            'accepted': accepted,
            'rejected': rejected,
            'acceptance_rate': accepted / total if total > 0 else 0,
            'average_score': avg_score
        }

# Test function (for development only)
def test_classifier():
    """Test the classifier with dummy data"""
    print("🧪 Testing classifier...")
    classifier = SariProductionClassifier()
    
    # This would work with real images
    # images = [Image.open("test1.jpg"), Image.open("test2.jpg")]
    # results = classifier.classify_images(images, ["test1.jpg", "test2.jpg"])
    # print(results)
    
    print("✅ Classifier loaded successfully!")

if __name__ == "__main__":
    test_classifier()
