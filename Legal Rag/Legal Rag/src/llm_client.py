"""
LLM Client module for legal analysis using LiquidAI/LFM2.5-1.2B-Instruct
"""

import logging
import json
import re
from typing import Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import get_config


logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with the LLM"""

    def __init__(self, config=None):
        """
        Initialize LLM client

        Args:
            config: Configuration object (uses global config if None)
        """
        self.config = config or get_config()
        self.model = None
        self.tokenizer = None
        self.device = self.config.get_device()

        self._load_model()

    def _load_model(self):
        """Load the LLM model and tokenizer"""
        model_name = self.config.llm.model_name
        logger.info(f"Loading LLM model: {model_name}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }

            if self.config.llm.load_in_8bit and self.device == "cuda":
                load_kwargs["load_in_8bit"] = True
                logger.info("Loading model in 8-bit mode")

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs
            )

            # Move to device
            if not self.config.llm.load_in_8bit:
                self.model = self.model.to(self.device)

            self.model.eval()  # Set to evaluation mode

            logger.info(f"Successfully loaded model on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def create_analysis_prompt(
        self,
        contract_clause: str,
        retrieved_laws: str,
        language: str = "ar"
    ) -> str:
        """
        Create prompt for legal analysis

        Args:
            contract_clause: Contract clause text
            retrieved_laws: Retrieved relevant laws
            language: Language code (ar for Arabic)

        Returns:
            Formatted prompt string
        """
        if language == "ar":
            prompt = f"""أنت خبير قانوني متخصص في قانون العمل المصري. مهمتك تحليل بنود العقد وإيجاد المخالفات بناءً على القوانين المصرية.

بند العقد:
{contract_clause}

القوانين ذات الصلة:
{retrieved_laws}

المطلوب:
1. حدد إذا كان هناك أي مخالفات لقانون العمل في هذا البند
2. صنف خطورة المخالفة (عالية/متوسطة/منخفضة)
3. اذكر المادة القانونية المخالفة
4. قدم توصية للإصلاح

قدم إجابتك بصيغة JSON كالتالي:
{{
  "has_violation": true أو false,
  "severity": "عالية" أو "متوسطة" أو "منخفضة",
  "violation_description": "وصف المخالفة",
  "violated_article": "رقم المادة المخالفة",
  "recommendation": "التوصية للإصلاح"
}}

الإجابة:"""
        else:
            # English fallback
            prompt = f"""You are a legal expert specialized in Egyptian labor law. Your task is to analyze contract clauses and identify violations.

Contract Clause:
{contract_clause}

Relevant Laws:
{retrieved_laws}

Required:
1. Determine if there are any labor law violations in this clause
2. Classify the severity (high/medium/low)
3. Specify the violated legal article
4. Provide a recommendation for correction

Provide your answer in JSON format:
{{
  "has_violation": true or false,
  "severity": "high" or "medium" or "low",
  "violation_description": "description of violation",
  "violated_article": "article number",
  "recommendation": "recommendation for correction"
}}

Answer:"""

        return prompt

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate text from prompt

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        max_new_tokens = max_new_tokens or self.config.llm.max_new_tokens
        temperature = temperature or self.config.llm.temperature

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=self.config.llm.top_p,
                    do_sample=self.config.llm.do_sample,
                    repetition_penalty=self.config.llm.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # Remove the prompt from output
            response = generated_text[len(prompt):].strip()

            return response

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""

    def analyze_clause(
        self,
        contract_clause: str,
        retrieved_laws: str
    ) -> Dict[str, Any]:
        """
        Analyze a contract clause for violations

        Args:
            contract_clause: Contract clause text
            retrieved_laws: Retrieved relevant laws

        Returns:
            Analysis result dictionary
        """
        # Create prompt
        prompt = self.create_analysis_prompt(contract_clause, retrieved_laws)

        # Generate response
        logger.info("Generating legal analysis...")
        response = self.generate(prompt)

        # Parse JSON from response
        analysis = self._parse_json_response(response)

        # Add raw response for debugging
        analysis["raw_response"] = response

        return analysis

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response

        Args:
            response: LLM response text

        Returns:
            Parsed dictionary
        """
        # Try to extract JSON from response
        try:
            # Look for JSON block
            json_match = re.search(r'\{.*\}', response, re.DOTALL)

            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                logger.info("Successfully parsed JSON response")
                return result
            else:
                logger.warning("No JSON found in response")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")

        # Return default structure if parsing fails
        return {
            "has_violation": None,
            "severity": "unknown",
            "violation_description": response,
            "violated_article": None,
            "recommendation": None,
            "parse_error": True
        }

    def batch_analyze(
        self,
        clauses_and_laws: list[tuple[str, str]]
    ) -> list[Dict[str, Any]]:
        """
        Analyze multiple clauses in batch

        Args:
            clauses_and_laws: List of (clause, laws) tuples

        Returns:
            List of analysis results
        """
        results = []

        for i, (clause, laws) in enumerate(clauses_and_laws, 1):
            logger.info(f"Analyzing clause {i}/{len(clauses_and_laws)}")
            result = self.analyze_clause(clause, laws)
            results.append(result)

        return results


if __name__ == "__main__":
    # Test LLM client
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Initializing LLM client...")
    client = LLMClient()

    # Test with sample clause and law
    test_clause = """
    يعمل الموظف 60 ساعة في الأسبوع مقابل راتب شهري قدره 2000 جنيه.
    """

    test_laws = """
    المادة 100 - قانون العمل رقم 14 لسنة 2025:
    ال يجوز تشغيل العامل تشغيال فعليا اكثر من ثماني ساعات في اليوم او ثماني واربعين ساعة في االسبوع.
    """

    print("\nTesting legal analysis...")
    print(f"Clause: {test_clause}")
    print(f"\nRelevant Law: {test_laws}")

    result = client.analyze_clause(test_clause, test_laws)

    print("\nAnalysis Result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
