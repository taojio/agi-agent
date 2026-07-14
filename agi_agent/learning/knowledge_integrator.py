import numpy as np
import torch
from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from ..config.settings import DEVICE


class IntegrationStrategy(Enum):
    MERGE = "merge"
    LINK = "link"
    TRANSFORM = "transform"
    ABSTRACT = "abstract"
    SPECIALIZE = "specialize"
    ANALOGIZE = "analogize"


class DomainType(Enum):
    GENERAL = "general"
    SCIENCE = "science"
    TECHNOLOGY = "technology"
    ART = "art"
    BUSINESS = "business"
    MEDICINE = "medicine"
    ENGINEERING = "engineering"
    SOCIAL = "social"
    MATHEMATICS = "mathematics"
    PHILOSOPHY = "philosophy"


class KnowledgeFragment:
    def __init__(self, fragment_id: str, content: str, domain: DomainType,
                 features: Optional[np.ndarray] = None, confidence: float = 0.5,
                 source: str = "unknown"):
        self.fragment_id = fragment_id
        self.content = content
        self.domain = domain
        self.features = features if features is not None else np.array([])
        self.confidence = confidence
        self.source = source
        self.integration_status = "raw"
        self.linked_fragments: List[str] = []
        self.timestamp = np.random.randint(1000000)
        self.metadata: Dict[str, Any] = {}

    def to_dict(self):
        return {
            "fragment_id": self.fragment_id,
            "content": self.content,
            "domain": self.domain.value,
            "confidence": self.confidence,
            "source": self.source,
            "integration_status": self.integration_status,
            "linked_fragments": self.linked_fragments,
            "metadata": self.metadata
        }


class IntegrationRecord:
    def __init__(self, record_id: str, strategy: IntegrationStrategy,
                 source_fragments: List[str], target_fragment: str,
                 confidence: float = 0.5, description: str = ""):
        self.record_id = record_id
        self.strategy = strategy
        self.source_fragments = source_fragments
        self.target_fragment = target_fragment
        self.confidence = confidence
        self.description = description
        self.timestamp = np.random.randint(1000000)

    def to_dict(self):
        return {
            "record_id": self.record_id,
            "strategy": self.strategy.value,
            "source_fragments": self.source_fragments,
            "target_fragment": self.target_fragment,
            "confidence": self.confidence,
            "description": self.description,
            "timestamp": self.timestamp
        }


class KnowledgeIntegrator:
    def __init__(self):
        self.fragments: Dict[str, KnowledgeFragment] = {}
        self.records: Dict[str, IntegrationRecord] = {}
        self.integration_history = deque(maxlen=500)
        self.domain_graph: Dict[str, List[str]] = {}
        self.similarity_threshold = 0.6
        self.confidence_threshold = 0.7

    def add_fragment(self, content: str, domain: DomainType,
                     features: Optional[np.ndarray] = None,
                     confidence: float = 0.5, source: str = "unknown") -> str:
        fragment_id = f"kfrag_{len(self.fragments) + 1}"
        fragment = KnowledgeFragment(fragment_id, content, domain, features, confidence, source)
        self.fragments[fragment_id] = fragment

        if domain.value not in self.domain_graph:
            self.domain_graph[domain.value] = []
        self.domain_graph[domain.value].append(fragment_id)

        self._auto_integrate(fragment_id)

        return fragment_id

    def _auto_integrate(self, new_fragment_id: str):
        new_fragment = self.fragments[new_fragment_id]
        candidates = self._find_integration_candidates(new_fragment)

        for candidate_id, similarity in candidates[:5]:
            if similarity > self.similarity_threshold:
                strategy = self._select_strategy(new_fragment, self.fragments[candidate_id], similarity)
                self.integrate(strategy, [new_fragment_id], candidate_id)

    def _find_integration_candidates(self, fragment: KnowledgeFragment) -> List[Tuple[str, float]]:
        candidates = []

        for frag_id, frag in self.fragments.items():
            if frag_id == fragment.fragment_id:
                continue

            similarity = self._compute_similarity(fragment, frag)
            if similarity > 0.3:
                candidates.append((frag_id, similarity))

        candidates.sort(key=lambda x: -x[1])
        return candidates

    def _compute_similarity(self, frag1: KnowledgeFragment, frag2: KnowledgeFragment) -> float:
        if frag1.features is None or frag2.features is None or len(frag1.features) == 0 or len(frag2.features) == 0:
            content_sim = self._compute_content_similarity(frag1.content, frag2.content)
            domain_sim = 1.0 if frag1.domain == frag2.domain else 0.3
            return (content_sim + domain_sim) / 2

        min_len = min(len(frag1.features), len(frag2.features))
        if min_len == 0:
            return 0.0

        f1 = np.array(frag1.features[:min_len]).flatten()
        f2 = np.array(frag2.features[:min_len]).flatten()
        feature_sim = np.dot(f1, f2) / (
            np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8
        )

        domain_sim = 1.0 if frag1.domain == frag2.domain else 0.3
        return (feature_sim + domain_sim) / 2

    def _compute_content_similarity(self, content1: str, content2: str) -> float:
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / max(len(union), 1)

    def _select_strategy(self, source: KnowledgeFragment, target: KnowledgeFragment,
                         similarity: float) -> IntegrationStrategy:
        if source.domain == target.domain:
            if similarity > 0.8:
                return IntegrationStrategy.MERGE
            elif similarity > 0.6:
                return IntegrationStrategy.LINK
            else:
                return IntegrationStrategy.TRANSFORM
        else:
            if similarity > 0.7:
                return IntegrationStrategy.ANALOGIZE
            elif similarity > 0.5:
                return IntegrationStrategy.ABSTRACT
            else:
                return IntegrationStrategy.LINK

    def integrate(self, strategy: IntegrationStrategy, source_ids: List[str],
                  target_id: str) -> Dict[str, Any]:
        if target_id not in self.fragments:
            return {"success": False, "message": "Target fragment not found"}

        record_id = f"int_{len(self.records) + 1}"

        target = self.fragments[target_id]
        sources = [self.fragments[s] for s in source_ids if s in self.fragments]

        if strategy == IntegrationStrategy.MERGE:
            result = self._merge_fragments(sources, target)
        elif strategy == IntegrationStrategy.LINK:
            result = self._link_fragments(sources, target)
        elif strategy == IntegrationStrategy.TRANSFORM:
            result = self._transform_fragments(sources, target)
        elif strategy == IntegrationStrategy.ABSTRACT:
            result = self._abstract_fragments(sources, target)
        elif strategy == IntegrationStrategy.SPECIALIZE:
            result = self._specialize_fragments(sources, target)
        elif strategy == IntegrationStrategy.ANALOGIZE:
            result = self._analogize_fragments(sources, target)
        else:
            return {"success": False, "message": "Unknown strategy"}

        record = IntegrationRecord(
            record_id, strategy, source_ids, target_id,
            result.get("confidence", 0.5), result.get("description", "")
        )
        self.records[record_id] = record
        self.integration_history.append(record.to_dict())

        return {
            "success": True,
            "record_id": record_id,
            "strategy": strategy.value,
            "result": result,
            "description": result.get("description", "")
        }

    def _merge_fragments(self, sources: List[KnowledgeFragment], target: KnowledgeFragment) -> Dict:
        merged_content = target.content
        merged_confidence = target.confidence

        for source in sources:
            if source.content not in target.content:
                merged_content += f"\n补充: {source.content}"
                merged_confidence = (merged_confidence + source.confidence) / 2

        target.content = merged_content
        target.confidence = merged_confidence
        target.integration_status = "merged"

        for source in sources:
            if source.fragment_id not in target.linked_fragments:
                target.linked_fragments.append(source.fragment_id)
            source.integration_status = "merged_into"

        return {
            "confidence": merged_confidence,
            "description": f"合并了 {len(sources)} 个片段到目标片段",
            "merged_content_length": len(merged_content)
        }

    def _link_fragments(self, sources: List[KnowledgeFragment], target: KnowledgeFragment) -> Dict:
        link_count = 0

        for source in sources:
            if source.fragment_id not in target.linked_fragments:
                target.linked_fragments.append(source.fragment_id)
                link_count += 1
            if target.fragment_id not in source.linked_fragments:
                source.linked_fragments.append(target.fragment_id)
                link_count += 1

        target.integration_status = "linked"

        return {
            "confidence": target.confidence,
            "description": f"建立了 {link_count} 条关联链接",
            "link_count": link_count
        }

    def _transform_fragments(self, sources: List[KnowledgeFragment], target: KnowledgeFragment) -> Dict:
        transformed_content = target.content

        for source in sources:
            if source.domain != target.domain:
                transformed_content += f"\n跨域视角({source.domain.value}): {source.content}"

        target.content = transformed_content
        target.integration_status = "transformed"

        for source in sources:
            if source.fragment_id not in target.linked_fragments:
                target.linked_fragments.append(source.fragment_id)

        return {
            "confidence": target.confidence * 0.9,
            "description": f"将 {len(sources)} 个跨域片段转化为目标领域视角",
            "transformed_domains": [s.domain.value for s in sources]
        }

    def _abstract_fragments(self, sources: List[KnowledgeFragment], target: KnowledgeFragment) -> Dict:
        all_contents = [target.content] + [s.content for s in sources]
        common_concepts = self._extract_common_concepts(all_contents)

        abstract_content = f"抽象概念: {', '.join(common_concepts)}\n\n原始内容:\n"
        for i, content in enumerate(all_contents):
            abstract_content += f"{i+1}. {content}\n"

        abstract_id = f"kfrag_{len(self.fragments) + 1}"
        abstract_fragment = KnowledgeFragment(
            abstract_id, abstract_content, DomainType.GENERAL,
            confidence=min(0.9, target.confidence + 0.1)
        )
        self.fragments[abstract_id] = abstract_fragment

        for source in sources + [target]:
            if abstract_id not in source.linked_fragments:
                source.linked_fragments.append(abstract_id)
            abstract_fragment.linked_fragments.append(source.fragment_id)

        return {
            "confidence": abstract_fragment.confidence,
            "description": f"从 {len(sources) + 1} 个片段中抽象出共同概念",
            "abstract_fragment_id": abstract_id,
            "common_concepts": common_concepts
        }

    def _extract_common_concepts(self, contents: List[str]) -> List[str]:
        all_words = []
        for content in contents:
            words = content.lower().split()
            all_words.extend(words)

        word_counts = {}
        for word in all_words:
            if len(word) > 2:
                word_counts[word] = word_counts.get(word, 0) + 1

        common_words = [w for w, c in word_counts.items() if c >= len(contents) * 0.5]
        return common_words[:10]

    def _specialize_fragments(self, sources: List[KnowledgeFragment], target: KnowledgeFragment) -> Dict:
        specialized_content = f"领域具体化({target.domain.value}):\n{target.content}\n\n具体应用:\n"

        for i, source in enumerate(sources):
            specialized_content += f"{i+1}. {source.domain.value}: {source.content}\n"

        target.content = specialized_content
        target.integration_status = "specialized"

        return {
            "confidence": target.confidence * 0.95,
            "description": f"将通用概念具体化为 {target.domain.value} 领域的应用",
            "specialized_domains": [s.domain.value for s in sources]
        }

    def _analogize_fragments(self, sources: List[KnowledgeFragment], target: KnowledgeFragment) -> Dict:
        analogy_content = f"类比推理:\n源领域({sources[0].domain.value}): {sources[0].content}\n\n目标领域({target.domain.value}): {target.content}\n\n类比关系:"

        for source in sources:
            analogy_content += f"\n- {source.domain.value} -> {target.domain.value}: 基于相似性"

        target.content = analogy_content
        target.integration_status = "analogized"

        for source in sources:
            if source.fragment_id not in target.linked_fragments:
                target.linked_fragments.append(source.fragment_id)

        return {
            "confidence": min(0.8, target.confidence + 0.1),
            "description": f"建立了 {len(sources)} 个跨域类比关系",
            "analogy_count": len(sources)
        }

    def distill_experience(self, experience_fragments: List[str],
                           target_domain: DomainType) -> Dict[str, Any]:
        if not experience_fragments:
            return {"success": False, "message": "No fragments provided"}

        fragments = [self.fragments[f] for f in experience_fragments if f in self.fragments]
        if not fragments:
            return {"success": False, "message": "No valid fragments"}

        distilled_content = f"经验蒸馏({target_domain.value}):\n\n"

        for i, fragment in enumerate(fragments):
            distilled_content += f"{i+1}. [{fragment.domain.value}] {fragment.content}\n"

        distilled_content += "\n关键经验总结:\n"
        key_insights = self._extract_key_insights(fragments)
        for j, insight in enumerate(key_insights):
            distilled_content += f"{j+1}. {insight}\n"

        distilled_id = f"kfrag_{len(self.fragments) + 1}"
        distilled_fragment = KnowledgeFragment(
            distilled_id, distilled_content, target_domain,
            confidence=min(0.95, np.mean([f.confidence for f in fragments]))
        )
        self.fragments[distilled_id] = distilled_fragment

        for fragment in fragments:
            if distilled_id not in fragment.linked_fragments:
                fragment.linked_fragments.append(distilled_id)
            distilled_fragment.linked_fragments.append(fragment.fragment_id)

        return {
            "success": True,
            "distilled_fragment_id": distilled_id,
            "confidence": distilled_fragment.confidence,
            "key_insights": key_insights,
            "source_count": len(fragments),
            "description": f"从 {len(fragments)} 个经验片段中蒸馏出 {len(key_insights)} 条关键经验"
        }

    def _extract_key_insights(self, fragments: List[KnowledgeFragment]) -> List[str]:
        insights = []
        all_content = " ".join([f.content for f in fragments])
        sentences = [s.strip() for s in all_content.split('\n') if s.strip()]

        for sentence in sentences[:10]:
            if len(sentence) > 10:
                insights.append(sentence)

        return insights[:5]

    def find_cross_domain_links(self, domain1: DomainType, domain2: DomainType,
                                max_results: int = 10) -> List[Dict]:
        fragments1 = self.domain_graph.get(domain1.value, [])
        fragments2 = self.domain_graph.get(domain2.value, [])

        cross_links = []
        for frag1_id in fragments1:
            frag1 = self.fragments[frag1_id]
            for frag2_id in fragments2:
                frag2 = self.fragments[frag2_id]
                if frag1_id in frag2.linked_fragments or frag2_id in frag1.linked_fragments:
                    similarity = self._compute_similarity(frag1, frag2)
                    cross_links.append({
                        "fragment1": frag1.to_dict(),
                        "fragment2": frag2.to_dict(),
                        "similarity": similarity
                    })

        cross_links.sort(key=lambda x: -x["similarity"])
        return cross_links[:max_results]

    def get_integration_summary(self) -> Dict[str, Any]:
        summary = {
            "total_fragments": len(self.fragments),
            "total_integrations": len(self.records),
            "integration_history_count": len(self.integration_history),
            "domain_distribution": {},
            "status_distribution": {},
            "avg_confidence": 0.0,
            "avg_links_per_fragment": 0.0
        }

        for domain in DomainType:
            count = len(self.domain_graph.get(domain.value, []))
            if count > 0:
                summary["domain_distribution"][domain.value] = count

        for frag in self.fragments.values():
            summary["status_distribution"][frag.integration_status] = \
                summary["status_distribution"].get(frag.integration_status, 0) + 1

        if self.fragments:
            summary["avg_confidence"] = float(np.mean([f.confidence for f in self.fragments.values()]))
            summary["avg_links_per_fragment"] = float(np.mean([len(f.linked_fragments) for f in self.fragments.values()]))

        return summary

    def get_fragment_details(self, fragment_id: str) -> Optional[Dict]:
        if fragment_id not in self.fragments:
            return None

        fragment = self.fragments[fragment_id]
        details = fragment.to_dict()

        linked_details = []
        for linked_id in fragment.linked_fragments[:5]:
            if linked_id in self.fragments:
                linked_details.append({
                    "fragment_id": linked_id,
                    "content": self.fragments[linked_id].content[:50],
                    "domain": self.fragments[linked_id].domain.value,
                    "confidence": self.fragments[linked_id].confidence
                })

        details["linked_fragments_detail"] = linked_details
        return details