"""
Multimodal utilities for image extraction and analysis from PDFs.
"""

import os
import io
import base64
import logging
from typing import List, Dict, Tuple, Optional, Any
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import cv2
from transformers import pipeline
import torch
from langchain.schema import Document
import requests

# Set environment variables to reduce memory usage
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

logger = logging.getLogger(__name__)


class ImageExtractor:
    """Extract images from PDF documents."""
    
    def __init__(self):
        self.supported_formats = {'.pdf'}
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract all images from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing image data and metadata
        """
        images = []
        
        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Get image list from page
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Convert to PIL Image
                        img_data = pix.tobytes("png")
                        pil_image = Image.open(io.BytesIO(img_data))
                        
                        # Get image metadata
                        img_info = {
                            'page': page_num + 1,
                            'index': img_index,
                            'width': img[2],
                            'height': img[3],
                            'image': pil_image,
                            'image_data': img_data,
                            'bbox': img[1],  # Bounding box on page
                            'xref': xref
                        }
                        
                        images.append(img_info)
                        pix = None  # Free memory
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                        continue
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extracting images from PDF {pdf_path}: {e}")
        
        return images
    
    def save_extracted_images(self, images: List[Dict[str, Any]], output_dir: str) -> List[str]:
        """
        Save extracted images to disk.
        
        Args:
            images: List of image dictionaries
            output_dir: Directory to save images
            
        Returns:
            List of saved image file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        for i, img_info in enumerate(images):
            try:
                filename = f"image_page{img_info['page']}_{i}.png"
                filepath = os.path.join(output_dir, filename)
                
                img_info['image'].save(filepath, 'PNG')
                saved_paths.append(filepath)
                
                # Add filepath to image info
                img_info['filepath'] = filepath
                
            except Exception as e:
                logger.error(f"Failed to save image {i}: {e}")
        
        return saved_paths


class ImageAnalyzer:
    """Lightweight image analyzer for PDF documents."""
    
    def __init__(self, use_heavy_models: bool = False):
        """
        Initialize the image analyzer.
        
        Args:
            use_heavy_models: If True, loads heavy vision models (may cause memory issues)
        """
        self.use_heavy_models = use_heavy_models
        self.device = "cpu"  # Force CPU to avoid memory issues
        
        # Initialize lightweight components only
        self.image_processor = None
        self.caption_model = None
        self.classification_model = None
        
        if use_heavy_models:
            try:
                self._load_heavy_models()
            except Exception as e:
                logger.warning(f"Failed to load heavy models: {e}. Using lightweight mode.")
                self.use_heavy_models = False
    
    def _load_heavy_models(self):
        """Load heavy vision models (may cause memory issues)."""
        try:
            from transformers import AutoProcessor, AutoModelForImageClassification, AutoModelForVision2Seq
            
            # Load image processor
            self.image_processor = AutoProcessor.from_pretrained(
                "microsoft/git-base-coco",
                use_fast=True
            )
            
            # Load captioning model
            self.caption_model = AutoModelForVision2Seq.from_pretrained(
                "microsoft/git-base-coco",
                torch_dtype="auto"
            ).to(self.device)
            
            # Load classification model
            self.classification_model = AutoModelForImageClassification.from_pretrained(
                "microsoft/resnet-50",
                torch_dtype="auto"
            ).to(self.device)
            
            logger.info(f"Image analyzer initialized with device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading heavy models: {e}")
            raise
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract images from PDF with basic analysis.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of image data with basic analysis
        """
        images = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                        else:  # CMYK: convert to RGB
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix1.tobytes("png")
                            pix1 = None
                        
                        # Convert to PIL Image
                        pil_image = Image.open(io.BytesIO(img_data))
                        
                        # Basic image analysis
                        image_info = {
                            'page': page_num + 1,
                            'index': img_index,
                            'size': pil_image.size,
                            'mode': pil_image.mode,
                            'format': pil_image.format,
                            'width': pil_image.width,
                            'height': pil_image.height,
                            'caption': self._generate_caption(pil_image),
                            'classification': self._classify_image(pil_image),
                            'data': img_data
                        }
                        
                        images.append(image_info)
                        
                        pix = None
                        
                    except Exception as e:
                        logger.warning(f"Error processing image {img_index} on page {page_num}: {e}")
                        continue
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extracting images from PDF: {e}")
        
        return images
    
    def _generate_caption(self, image: Image.Image) -> str:
        """Generate caption for image (lightweight version)."""
        if not self.use_heavy_models or self.caption_model is None:
            # Return basic image description
            return f"Image ({image.width}x{image.height}, {image.mode})"
        
        try:
            import torch
            
            # Process image
            inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                generated_ids = self.caption_model.generate(
                    pixel_values=inputs.pixel_values,
                    max_length=50,
                    num_beams=4,
                    return_dict_in_generate=True
                ).sequences
            
            caption = self.image_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return caption.strip()
            
        except Exception as e:
            logger.warning(f"Error generating caption: {e}")
            return f"Image ({image.width}x{image.height}, {image.mode})"
    
    def _classify_image(self, image: Image.Image) -> str:
        """Classify image content (lightweight version)."""
        if not self.use_heavy_models or self.classification_model is None:
            # Return basic classification based on image properties
            if image.width > image.height:
                return "landscape"
            elif image.height > image.width:
                return "portrait"
            else:
                return "square"
        
        try:
            import torch
            
            # Process image
            inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
            
            # Get classification
            with torch.no_grad():
                outputs = self.classification_model(**inputs)
                logits = outputs.logits
                predicted_class_id = logits.argmax(-1).item()
                
                # Get class name
                class_name = self.classification_model.config.id2label[predicted_class_id]
                return class_name
            
        except Exception as e:
            logger.warning(f"Error classifying image: {e}")
            return "unknown"


class MultimodalDocumentProcessor:
    """Process documents with both text and image content."""
    
    def __init__(self, image_output_dir: str = "extracted_images", enable_vision_models: bool = False):
        """
        Initialize the multimodal document processor.
        
        Args:
            image_output_dir: Directory to save extracted images
            enable_vision_models: Whether to enable heavy vision models (can cause memory issues)
        """
        self.image_extractor = ImageExtractor()
        self.image_analyzer = ImageAnalyzer(use_heavy_models=enable_vision_models)
        self.image_output_dir = image_output_dir
    
    def process_pdf_with_images(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a PDF document, extracting both text and images.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing text content and image analysis
        """
        results = {
            'text_content': None,
            'images': [],
            'image_analysis': [],
            'multimodal_summary': None,
            'error': None
        }
        
        try:
            # Extract images from PDF
            images = self.image_extractor.extract_images_from_pdf(pdf_path)
            
            if images:
                # Save images to disk
                saved_paths = self.image_extractor.save_extracted_images(
                    images, 
                    os.path.join(self.image_output_dir, os.path.basename(pdf_path))
                )
                
                # Analyze images
                image_analysis = self._analyze_images(images)
                
                results['images'] = images
                results['image_analysis'] = image_analysis
                
                # Create multimodal summary
                results['multimodal_summary'] = self._create_multimodal_summary(image_analysis)
            
        except Exception as e:
            results['error'] = str(e)
            logger.error(f"Error processing PDF {pdf_path}: {e}")
        
        return results
    
    def _analyze_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze multiple images.
        
        Args:
            images: List of image dictionaries
            
        Returns:
            List of analysis results
        """
        results = []
        
        for img_info in images:
            analysis = self._analyze_image(img_info['image'])
            
            # Combine image info with analysis
            combined_result = {
                **img_info,
                'analysis': analysis
            }
            
            results.append(combined_result)
        
        return results
    
    def _analyze_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze a single image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary containing analysis results
        """
        results = {
            'caption': None,
            'classification': None,
            'text_content': None,
            'error': None
        }
        
        try:
            # Generate caption only if vision models are enabled
            if self.image_analyzer.use_heavy_models:
                caption_result = self._generate_caption(image)
                results['caption'] = caption_result
            else:
                results['caption'] = "Image analysis disabled (lightweight mode)"
            
            # Classify image only if vision models are enabled
            if self.image_analyzer.use_heavy_models:
                classification_result = self._classify_image(image)
                results['classification'] = classification_result
            else:
                results['classification'] = "Image classification disabled (lightweight mode)"
            
            # Extract text from image (OCR-like functionality)
            results['text_content'] = self._extract_text_from_image(image)
            
        except Exception as e:
            results['error'] = str(e)
            logger.error(f"Error analyzing image: {e}")
        
        return results
    
    def _generate_caption(self, image: Image.Image) -> str:
        """Generate caption for image (lightweight version)."""
        if not self.image_analyzer.use_heavy_models or self.image_analyzer.caption_model is None:
            # Return basic image description
            return f"Image ({image.width}x{image.height}, {image.mode})"
        
        try:
            import torch
            
            # Process image
            inputs = self.image_analyzer.image_processor(images=image, return_tensors="pt").to(self.image_analyzer.device)
            
            # Generate caption
            with torch.no_grad():
                generated_ids = self.image_analyzer.caption_model.generate(
                    pixel_values=inputs.pixel_values,
                    max_length=50,
                    num_beams=4,
                    return_dict_in_generate=True
                ).sequences
            
            caption = self.image_analyzer.image_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return caption.strip()
            
        except Exception as e:
            logger.warning(f"Error generating caption: {e}")
            return f"Image ({image.width}x{image.height}, {image.mode})"
    
    def _classify_image(self, image: Image.Image) -> str:
        """Classify image content (lightweight version)."""
        if not self.image_analyzer.use_heavy_models or self.image_analyzer.classification_model is None:
            # Return basic classification based on image properties
            if image.width > image.height:
                return "landscape"
            elif image.height > image.width:
                return "portrait"
            else:
                return "square"
        
        try:
            import torch
            
            # Process image
            inputs = self.image_analyzer.image_processor(images=image, return_tensors="pt").to(self.image_analyzer.device)
            
            # Get classification
            with torch.no_grad():
                outputs = self.image_analyzer.classification_model(**inputs)
                logits = outputs.logits
                predicted_class_id = logits.argmax(-1).item()
                
                # Get class name
                class_name = self.image_analyzer.classification_model.config.id2label[predicted_class_id]
                return class_name
            
        except Exception as e:
            logger.warning(f"Error classifying image: {e}")
            return "unknown"
    
    def _extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extract text content from image using OCR-like approach.
        This is a simplified version - for production, consider using Tesseract OCR.
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text (placeholder for now)
        """
        # Placeholder for OCR functionality
        # In a full implementation, you would use Tesseract or similar OCR
        return "Text extraction not implemented yet"
    
    def _create_multimodal_summary(self, image_analysis: List[Dict[str, Any]]) -> str:
        """
        Create a summary of image content for inclusion in document chunks.
        
        Args:
            image_analysis: List of image analysis results
            
        Returns:
            Summary string
        """
        if not image_analysis:
            return ""
        
        summary_parts = ["\n\n=== IMAGE CONTENT ===\n"]
        
        for i, analysis in enumerate(image_analysis):
            page_num = analysis.get('page', i + 1)
            caption = analysis.get('caption', 'No caption available')
            classification = analysis.get('classification', [])
            
            summary_parts.append(f"Page {page_num}, Image {i + 1}:")
            summary_parts.append(f"  Caption: {caption}")
            
            if classification:
                top_class = classification[0]
                summary_parts.append(f"  Classification: {top_class}")
            
            summary_parts.append("")
        
        return "\n".join(summary_parts)
    
    def create_multimodal_documents(self, pdf_path: str, text_chunks: List[Document]) -> List[Document]:
        """
        Create multimodal documents by combining text chunks with image analysis.
        
        Args:
            pdf_path: Path to the PDF file
            text_chunks: List of text document chunks
            
        Returns:
            List of multimodal document chunks
        """
        # Process PDF for images
        multimodal_data = self.process_pdf_with_images(pdf_path)
        
        if not multimodal_data.get('image_analysis'):
            return text_chunks
        
        # Create image summary
        image_summary = multimodal_data.get('multimodal_summary', "")
        
        # Add image content to each text chunk
        multimodal_chunks = []
        
        for chunk in text_chunks:
            # Create new chunk with image content
            multimodal_chunk = Document(
                page_content=chunk.page_content + image_summary,
                metadata={
                    **chunk.metadata,
                    'has_images': True,
                    'image_count': len(multimodal_data.get('images', [])),
                    'multimodal': True
                }
            )
            
            multimodal_chunks.append(multimodal_chunk)
        
        return multimodal_chunks


def encode_image_to_base64(image: Image.Image) -> str:
    """
    Encode PIL image to base64 string for API calls.
    
    Args:
        image: PIL Image object
        
    Returns:
        Base64 encoded string
    """
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


def decode_base64_to_image(base64_string: str) -> Image.Image:
    """
    Decode base64 string to PIL image.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        PIL Image object
    """
    img_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_data))


def get_image_statistics(image: Image.Image) -> Dict[str, Any]:
    """
    Get basic statistics about an image.
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary with image statistics
    """
    return {
        'size': image.size,
        'mode': image.mode,
        'format': image.format,
        'width': image.width,
        'height': image.height,
        'aspect_ratio': image.width / image.height if image.height > 0 else 0
    }


def resize_image_for_analysis(image: Image.Image, max_size: int = 512) -> Image.Image:
    """
    Resize image for analysis while maintaining aspect ratio.
    
    Args:
        image: PIL Image object
        max_size: Maximum dimension size
        
    Returns:
        Resized PIL Image object
    """
    # Calculate new size maintaining aspect ratio
    ratio = min(max_size / image.width, max_size / image.height)
    new_size = (int(image.width * ratio), int(image.height * ratio))
    
    return image.resize(new_size, Image.Resampling.LANCZOS)


def extract_images_from_pdf(pdf_path: str, use_heavy_models: bool = False) -> List[Dict[str, Any]]:
    """
    Extract and analyze images from PDF.
    
    Args:
        pdf_path: Path to the PDF file
        use_heavy_models: Whether to use heavy vision models
        
    Returns:
        List of image data with analysis
    """
    analyzer = ImageAnalyzer(use_heavy_models=use_heavy_models)
    return analyzer.extract_images_from_pdf(pdf_path)


def analyze_standalone_image(image_path: str, use_heavy_models: bool = False) -> str:
    """
    Analyze a standalone image file.
    
    Args:
        image_path: Path to the image file
        use_heavy_models: Whether to use heavy vision models
        
    Returns:
        Analysis summary string
    """
    try:
        from PIL import Image
        
        # Load image
        image = Image.open(image_path)
        
        # Get basic image information
        image_info = {
            'width': image.width,
            'height': image.height,
            'mode': image.mode,
            'format': image.format,
            'size_bytes': os.path.getsize(image_path)
        }
        
        # Create analyzer
        analyzer = ImageAnalyzer(use_heavy_models=use_heavy_models)
        
        # Analyze image
        caption = analyzer._generate_caption(image)
        classification = analyzer._classify_image(image)
        
        # Create summary
        summary = f"""
Image Analysis Results:
- File: {os.path.basename(image_path)}
- Dimensions: {image_info['width']}x{image_info['height']} pixels
- Format: {image_info['format']} ({image_info['mode']})
- Size: {image_info['size_bytes']} bytes
- Caption: {caption}
- Classification: {classification}
        """.strip()
        
        return summary
        
    except Exception as e:
        logger.error(f"Error analyzing standalone image {image_path}: {e}")
        return f"Error analyzing image: {str(e)}"


def get_image_analysis_summary(image_path: str, use_heavy_models: bool = False) -> str:
    """
    Get a summary of image analysis.
    
    Args:
        image_path: Path to the image file (PDF or standalone image)
        use_heavy_models: Whether to use heavy vision models
        
    Returns:
        Summary string of image analysis
    """
    try:
        # Check if it's a PDF or standalone image
        if image_path.lower().endswith('.pdf'):
            # Extract images from PDF
            images = extract_images_from_pdf(image_path, use_heavy_models)
            
            if not images:
                return "No images found in the document."
            
            summary_parts = [f"Found {len(images)} image(s) in the document:"]
            
            for i, img in enumerate(images, 1):
                summary_parts.append(
                    f"  {i}. Page {img['page']}: {img['caption']} "
                    f"({img['width']}x{img['height']}, {img['classification']})"
                )
            
            return "\n".join(summary_parts)
        else:
            # Analyze standalone image
            return analyze_standalone_image(image_path, use_heavy_models)
        
    except Exception as e:
        logger.error(f"Error generating image summary: {e}")
        return "Error analyzing images in the document."


def check_multimodal_capabilities() -> bool:
    """Check if multimodal capabilities are available."""
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        return True
    except ImportError:
        return False 