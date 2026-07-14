import numpy as np
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


class GeneTemplate:
    def __init__(self, name: str, gene_type: str = "float",
                 min_value: Any = None, max_value: Any = None,
                 description: str = "", default_value: Any = None):
        self.name = name
        self.gene_type = gene_type
        self.min_value = min_value
        self.max_value = max_value
        self.description = description
        self.default_value = default_value
        self.usage_count: int = 0
        self.success_count: int = 0

    def create_gene(self, value: Any = None) -> Dict[str, Any]:
        self.usage_count += 1
        
        if value is not None:
            return {
                "name": self.name,
                "value": value,
                "min_value": self.min_value,
                "max_value": self.max_value,
                "gene_type": self.gene_type
            }
        
        if self.default_value is not None:
            return {
                "name": self.name,
                "value": self.default_value,
                "min_value": self.min_value,
                "max_value": self.max_value,
                "gene_type": self.gene_type
            }
        
        if self.gene_type == "float":
            value = np.random.uniform(self.min_value, self.max_value) if self.min_value is not None else np.random.random()
        elif self.gene_type == "integer":
            value = np.random.randint(self.min_value, self.max_value + 1) if self.min_value is not None else np.random.randint(0, 10)
        elif self.gene_type == "categorical":
            categories = self.max_value if isinstance(self.max_value, list) else []
            value = np.random.choice(categories) if categories else ""
        else:
            value = np.random.random()
        
        return {
            "name": self.name,
            "value": value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "gene_type": self.gene_type
        }

    def record_success(self):
        self.success_count += 1

    def get_effectiveness(self) -> float:
        if self.usage_count == 0:
            return 0.5
        return self.success_count / self.usage_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "gene_type": self.gene_type,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "description": self.description,
            "default_value": self.default_value,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "effectiveness": self.get_effectiveness()
        }


class GeneLibrary:
    def __init__(self):
        self.templates: Dict[str, GeneTemplate] = {}
        self.categories: Dict[str, List[str]] = {}

    def add_template(self, template: GeneTemplate, category: str = "default"):
        self.templates[template.name] = template
        
        if category not in self.categories:
            self.categories[category] = []
        if template.name not in self.categories[category]:
            self.categories[category].append(template.name)

    def get_template(self, name: str) -> Optional[GeneTemplate]:
        return self.templates.get(name)

    def get_templates_by_category(self, category: str) -> List[GeneTemplate]:
        if category not in self.categories:
            return []
        return [self.templates[name] for name in self.categories[category]]

    def create_template(self, name: str, gene_type: str = "float",
                       min_value: Any = None, max_value: Any = None,
                       description: str = "", default_value: Any = None,
                       category: str = "default") -> GeneTemplate:
        template = GeneTemplate(name, gene_type, min_value, max_value, description, default_value)
        self.add_template(template, category)
        return template

    def remove_template(self, name: str):
        if name in self.templates:
            template = self.templates[name]
            for category in self.categories.values():
                if name in category:
                    category.remove(name)
            del self.templates[name]

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_templates": len(self.templates),
            "categories": {cat: len(names) for cat, names in self.categories.items()},
            "templates": {name: template.to_dict() for name, template in self.templates.items()}
        }


class GenePool:
    def __init__(self):
        self.library = GeneLibrary()
        self.active_genes: Dict[str, List[Dict[str, Any]]] = {}
        self.gene_history: deque = deque(maxlen=200)

    def create_gene(self, template_name: str, value: Any = None) -> Optional[Dict[str, Any]]:
        template = self.library.get_template(template_name)
        if not template:
            return None
        
        gene = template.create_gene(value)
        
        if template_name not in self.active_genes:
            self.active_genes[template_name] = []
        self.active_genes[template_name].append(gene)
        
        self.gene_history.append({
            "template_name": template_name,
            "gene": gene,
            "timestamp": np.random.randint(1000000)
        })
        
        return gene

    def create_gene_set(self, template_names: List[str]) -> List[Dict[str, Any]]:
        genes = []
        for name in template_names:
            gene = self.create_gene(name)
            if gene:
                genes.append(gene)
        return genes

    def get_gene_variants(self, template_name: str) -> List[Dict[str, Any]]:
        return self.active_genes.get(template_name, [])

    def record_gene_success(self, template_name: str):
        template = self.library.get_template(template_name)
        if template:
            template.record_success()

    def prune_low_effectiveness(self, threshold: float = 0.3):
        low_effectiveness = []
        for name, template in self.library.templates.items():
            if template.get_effectiveness() < threshold:
                low_effectiveness.append(name)
        
        for name in low_effectiveness:
            self.library.remove_template(name)
        
        return low_effectiveness

    def suggest_genes(self, category: str = "default", limit: int = 5) -> List[Dict[str, Any]]:
        templates = self.library.get_templates_by_category(category)
        templates.sort(key=lambda t: t.get_effectiveness(), reverse=True)
        
        suggestions = []
        for template in templates[:limit]:
            suggestions.append({
                "template_name": template.name,
                "description": template.description,
                "effectiveness": template.get_effectiveness(),
                "usage_count": template.usage_count
            })
        
        return suggestions

    def generate_random_genome(self, template_names: List[str]) -> List[Dict[str, Any]]:
        genes = []
        for name in template_names:
            gene = self.create_gene(name)
            if gene:
                genes.append(gene)
        return genes

    def get_pool_summary(self) -> Dict[str, Any]:
        return {
            "library": self.library.get_summary(),
            "active_gene_count": sum(len(genes) for genes in self.active_genes.values()),
            "active_genes_by_template": {name: len(genes) for name, genes in self.active_genes.items()},
            "total_gene_creations": len(self.gene_history)
        }