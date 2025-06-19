import ast
import asyncio
import hashlib
from difflib import SequenceMatcher
from typing import List, Tuple, Union

import numpy as np
import openai
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


class Metrics:
    """Comprehensive metrics for food analysis evaluation."""

    _embedding_cache = {}
    _openai_client = None

    @classmethod
    async def close_openai_client(cls):
        """Close the OpenAI client."""
        if cls._openai_client is not None:
            await cls._openai_client.close()
            cls._openai_client = None

    @staticmethod
    def _normalize_ingredient_list(ingredients: Union[str, list, None]) -> List[dict]:
        """
        Centralized cleanup function.
        Safely parses ingredient data which might be a string representation of a list.
        
        Args:
            ingredients: Raw ingredient data
            
        Returns:
            Normalized list of ingredient dictionaries
        """
        if not ingredients:
            return []
        if isinstance(ingredients, list):
            return ingredients
        if isinstance(ingredients, str):
            try:
                parsed = ast.literal_eval(ingredients)
                return parsed if isinstance(parsed, list) else []
            except (ValueError, SyntaxError):
                return []
        return []

    @staticmethod
    async def get_embedding(text, model="text-embedding-3-small"):
        """
        Get OpenAI embedding with caching.
        
        Args:
            text: Text to embed
            model: OpenAI embedding model to use
            
        Returns:
            Embedding vector
        """
        cache_key = hashlib.md5(f"{text}_{model}".encode()).hexdigest()
        if cache_key in Metrics._embedding_cache:
            return Metrics._embedding_cache[cache_key]
        try:
            if Metrics._openai_client is None:
                Metrics._openai_client = openai.AsyncOpenAI()
            response = await Metrics._openai_client.embeddings.create(model=model, input=text.strip())
            embedding = response.data[0].embedding
            Metrics._embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            print(f"Error getting embedding for '{text}': {e}")
            return [0.0] * 1536

    @staticmethod
    async def semantic_ingredient_match_embeddings(
        gt_ingredients, pred_ingredients, threshold=0.75
    ) -> Tuple[float, List[dict]]:
        """
        Semantic ingredient matching using OpenAI embeddings and cosine similarity.
        
        Args:
            gt_ingredients: Ground truth ingredients
            pred_ingredients: Predicted ingredients
            threshold: Similarity threshold for matching
            
        Returns:
            Tuple of (recall score, match details)
        """
        gt_ingredients = Metrics._normalize_ingredient_list(gt_ingredients)
        pred_ingredients = Metrics._normalize_ingredient_list(pred_ingredients)

        def normalize_name(item):
            return str(item.get("name", "")).lower().strip()

        gt_names = [normalize_name(x) for x in gt_ingredients if normalize_name(x)]
        pred_names = [normalize_name(x) for x in pred_ingredients if normalize_name(x)]

        if not gt_names and not pred_names:
            return 1.0, []
        if not gt_names or not pred_names:
            return 0.0, []

        gt_embeddings = await asyncio.gather(
            *(Metrics.get_embedding(name) for name in gt_names)
        )
        pred_embeddings = await asyncio.gather(
            *(Metrics.get_embedding(name) for name in pred_names)
        )

        gt_embeddings = [emb for emb in gt_embeddings if emb and len(emb) > 1]
        pred_embeddings = [emb for emb in pred_embeddings if emb and len(emb) > 1]

        if not gt_embeddings or not pred_embeddings:
            return 0.0, []

        similarity_matrix = cosine_similarity(gt_embeddings, pred_embeddings)
        cost_matrix = 1 - similarity_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches = 0
        match_details = []
        for row_idx, col_idx in zip(row_indices, col_indices):
            similarity = similarity_matrix[row_idx, col_idx]
            if similarity >= threshold:
                matches += 1
                match_details.append(
                    {
                        "gt_ingredient": gt_names[row_idx],
                        "pred_ingredient": pred_names[col_idx],
                        "similarity": similarity,
                    }
                )

        recall = matches / len(gt_names)
        return recall, match_details

    @staticmethod
    def semantic_ingredient_match(gt_ingredients, pred_ingredients, threshold=0.7):
        """
        Fallback method using string similarity.
        
        Args:
            gt_ingredients: Ground truth ingredients
            pred_ingredients: Predicted ingredients
            threshold: Similarity threshold for matching
            
        Returns:
            Recall score
        """
        gt_ingredients = Metrics._normalize_ingredient_list(gt_ingredients)
        pred_ingredients = Metrics._normalize_ingredient_list(pred_ingredients)

        def normalize_name(item):
            name_str = item.get("name", "") if isinstance(item, dict) else item
            return str(name_str).lower().strip().replace("-", " ").replace("_", " ")

        gt_names = [normalize_name(x) for x in gt_ingredients if normalize_name(x)]
        pred_names = [normalize_name(x) for x in pred_ingredients if normalize_name(x)]

        if not gt_names:
            return 1.0 if not pred_names else 0.0

        matches = 0
        for gt_name in gt_names:
            if pred_names:
                best_match = max(
                    SequenceMatcher(None, gt_name, pred_name).ratio()
                    for pred_name in pred_names
                )
                if best_match >= threshold:
                    matches += 1

        return matches / len(gt_names)

    @staticmethod
    async def semantic_precision_score(
        gt_ingredients, pred_ingredients, threshold=0.75
    ):
        """
        Calculates precision based on semantic matches from embeddings.
        
        Args:
            gt_ingredients: Ground truth ingredients
            pred_ingredients: Predicted ingredients
            threshold: Similarity threshold for matching
            
        Returns:
            Precision score
        """
        gt_list = Metrics._normalize_ingredient_list(gt_ingredients)
        pred_list = Metrics._normalize_ingredient_list(pred_ingredients)

        def normalize_name(item):
            return str(item.get("name", "")).lower().strip()

        gt_names = [normalize_name(x) for x in gt_list if normalize_name(x)]
        pred_names = [normalize_name(x) for x in pred_list if normalize_name(x)]

        if not pred_names:
            return 1.0
        if not gt_names:
            return 0.0

        _, match_details = await Metrics.semantic_ingredient_match_embeddings(
            gt_list, pred_list, threshold
        )

        matches = len(match_details)
        precision = matches / len(pred_names)
        return precision

    @staticmethod
    async def semantic_f1_score(gt_ingredients, pred_ingredients, threshold=0.75):
        """
        Calculates F1 score based on semantic matches.
        
        Args:
            gt_ingredients: Ground truth ingredients
            pred_ingredients: Predicted ingredients
            threshold: Similarity threshold for matching
            
        Returns:
            F1 score
        """
        gt_list = Metrics._normalize_ingredient_list(gt_ingredients)
        pred_list = Metrics._normalize_ingredient_list(pred_ingredients)

        def normalize_name(item):
            return str(item.get("name", "")).lower().strip()

        gt_names = [normalize_name(x) for x in gt_list if normalize_name(x)]
        pred_names = [normalize_name(x) for x in pred_list if normalize_name(x)]

        if not gt_names and not pred_names:
            return 1.0
        if not gt_names or not pred_names:
            return 0.0

        _, match_details = await Metrics.semantic_ingredient_match_embeddings(
            gt_list, pred_list, threshold
        )

        matches = len(match_details)

        if not pred_names or not gt_names:
            return 0.0

        precision = matches / len(pred_names)
        recall = matches / len(gt_names)

        if precision + recall == 0:
            return 0.0

        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    @staticmethod
    async def meal_name_similarity(gt_name: str, pred_name: str) -> float:
        """
        Calculates cosine similarity between the embeddings of two meal names.
        
        Args:
            gt_name: Ground truth meal name
            pred_name: Predicted meal name
            
        Returns:
            Cosine similarity score
        """
        gt_name = str(gt_name or "").strip()
        pred_name = str(pred_name or "").strip()

        if not pred_name:
            return 0.0

        gt_embedding_list, pred_embedding_list = await asyncio.gather(
            Metrics.get_embedding(gt_name), Metrics.get_embedding(pred_name)
        )

        gt_embedding = np.array(gt_embedding_list).reshape(1, -1)
        pred_embedding = np.array(pred_embedding_list).reshape(1, -1)

        return cosine_similarity(gt_embedding, pred_embedding)[0][0]

    @staticmethod
    def macro_wMAPE(gt_mac: dict, pred_mac: dict):
        """
        Calculates Weighted Mean Absolute Percentage Error over the four main macros.
        
        Args:
            gt_mac: Ground truth macros dictionary
            pred_mac: Predicted macros dictionary
            
        Returns:
            wMAPE percentage
        """
        keys = ["calories", "carbs", "protein", "fat"]

        absolute_errors = sum(abs(gt_mac.get(k, 0) - pred_mac.get(k, 0)) for k in keys)
        sum_of_actuals = sum(abs(gt_mac.get(k, 0)) for k in keys)

        if sum_of_actuals == 0:
            return 0.0 if absolute_errors == 0 else 100.0

        return (absolute_errors / sum_of_actuals) * 100

    @staticmethod
    def macro_percentage_error(gt_mac, pred_mac):
        """
        Calculate percentage error for each macro.
        
        Args:
            gt_mac: Ground truth macros dictionary
            pred_mac: Predicted macros dictionary
            
        Returns:
            Dictionary of percentage errors for each macro
        """
        keys = ["calories", "carbs", "protein", "fat"]
        errors = {}
        for key in keys:
            gt_val = gt_mac.get(key, 0)
            pred_val = pred_mac.get(key, 0)
            if gt_val > 0:
                errors[key] = abs(gt_val - pred_val) / gt_val * 100
            else:
                errors[key] = 0 if pred_val == 0 else 100
        return errors

    @staticmethod
    def ingredient_count_accuracy(gt_ingredients, pred_ingredients):
        """
        How well does the model predict the number of ingredients?
        
        Args:
            gt_ingredients: Ground truth ingredients
            pred_ingredients: Predicted ingredients
            
        Returns:
            Accuracy score between 0 and 1
        """
        gt_count = len(Metrics._normalize_ingredient_list(gt_ingredients))
        pred_count = len(Metrics._normalize_ingredient_list(pred_ingredients))
        if gt_count == 0 and pred_count == 0:
            return 1.0
        if gt_count == 0:
            return 0.0
        return 1 - abs(gt_count - pred_count) / gt_count