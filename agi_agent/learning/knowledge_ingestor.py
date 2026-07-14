import json
import csv
import numpy as np
from collections import deque


class StructuredKnowledgeIngestor:
    def __init__(self, knowledge_graph, causal_reasoner):
        self.knowledge_graph = knowledge_graph
        self.causal_reasoner = causal_reasoner
        self.ingestion_history = deque(maxlen=50)
        self.supported_formats = ['json', 'csv', 'ontology', 'rules']
        self.enabled = True
    
    def ingest(self, source, format_type='json'):
        if not self.enabled:
            return {'success': False, 'message': 'Ingestor disabled'}
        
        if format_type not in self.supported_formats:
            return {'success': False, 'message': f'Unsupported format: {format_type}'}
        
        try:
            if format_type == 'json':
                result = self._ingest_json(source)
            elif format_type == 'csv':
                result = self._ingest_csv(source)
            elif format_type == 'ontology':
                result = self._ingest_ontology(source)
            elif format_type == 'rules':
                result = self._ingest_rules(source)
            else:
                return {'success': False, 'message': f'Unknown format: {format_type}'}
            
            self.ingestion_history.append({
                'format': format_type,
                'success': result['success'],
                'count': result.get('count', 0),
                'timestamp': np.random.randint(1000000)
            })
            
            return result
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def _ingest_json(self, data):
        if isinstance(data, str):
            data = json.loads(data)
        
        nodes_added = 0
        edges_added = 0
        
        if 'nodes' in data:
            for node in data['nodes']:
                node_id = node.get('id', f"node_{np.random.randint(1000000)}")
                properties = node.get('properties', {})
                self.knowledge_graph.add_node(node_id, properties)
                nodes_added += 1
        
        if 'edges' in data:
            for edge in data['edges']:
                source = edge.get('source')
                target = edge.get('target')
                relation = edge.get('relation', 'related_to')
                if source and target:
                    self.knowledge_graph.add_edge(source, target, relation)
                    edges_added += 1
        
        if 'causal_relations' in data:
            for rel in data['causal_relations']:
                cause = rel.get('cause')
                effect = rel.get('effect')
                strength = rel.get('strength', 0.5)
                if cause and effect:
                    self._add_causal_relation(cause, effect, strength)
        
        return {
            'success': True,
            'count': nodes_added + edges_added,
            'nodes_added': nodes_added,
            'edges_added': edges_added
        }
    
    def _ingest_csv(self, file_path):
        nodes_added = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_id = row.get('id', f"csv_node_{np.random.randint(1000000)}")
                properties = {k: v for k, v in row.items() if k != 'id'}
                self.knowledge_graph.add_node(node_id, properties)
                nodes_added += 1
        
        return {
            'success': True,
            'count': nodes_added,
            'nodes_added': nodes_added
        }
    
    def _ingest_ontology(self, data):
        if isinstance(data, str):
            data = json.loads(data)
        
        nodes_added = 0
        
        if 'concepts' in data:
            for concept in data['concepts']:
                concept_id = concept.get('id', f"concept_{np.random.randint(1000000)}")
                properties = {
                    'type': 'concept',
                    'label': concept.get('label', ''),
                    'definition': concept.get('definition', ''),
                    'parent': concept.get('parent', None)
                }
                self.knowledge_graph.add_node(concept_id, properties)
                nodes_added += 1
                
                parent = concept.get('parent')
                if parent:
                    self.knowledge_graph.add_edge(parent, concept_id, 'is_a')
        
        if 'instances' in data:
            for instance in data['instances']:
                instance_id = instance.get('id', f"instance_{np.random.randint(1000000)}")
                concept_type = instance.get('type')
                properties = {
                    'type': 'instance',
                    'label': instance.get('label', '')
                }
                self.knowledge_graph.add_node(instance_id, properties)
                nodes_added += 1
                
                if concept_type:
                    self.knowledge_graph.add_edge(instance_id, concept_type, 'instance_of')
        
        return {
            'success': True,
            'count': nodes_added
        }
    
    def _ingest_rules(self, rules):
        if isinstance(rules, str):
            rules = json.loads(rules)
        
        rules_added = 0
        
        for rule in rules:
            rule_id = rule.get('id', f"rule_{np.random.randint(1000000)}")
            condition = rule.get('condition', {})
            action = rule.get('action', {})
            
            rule_data = {
                'type': 'rule',
                'condition': condition,
                'action': action,
                'priority': rule.get('priority', 1.0)
            }
            
            self.knowledge_graph.add_node(rule_id, rule_data)
            rules_added += 1
            
            if 'causal_effect' in rule:
                cause = rule.get('cause')
                effect = rule.get('effect')
                strength = rule.get('causal_effect', 0.5)
                if cause and effect:
                    self._add_causal_relation(cause, effect, strength)
        
        return {
            'success': True,
            'count': rules_added
        }
    
    def _add_causal_relation(self, cause, effect, strength):
        cause_node = self._resolve_node(cause)
        effect_node = self._resolve_node(effect)
        
        if cause_node is not None and effect_node is not None:
            self.causal_reasoner.causal_graph.add_edge(cause_node, effect_node, strength)
    
    def _resolve_node(self, node_identifier):
        if isinstance(node_identifier, int):
            return node_identifier
        
        if hasattr(self.knowledge_graph, 'nodes'):
            for idx, (node_id, data) in enumerate(self.knowledge_graph.nodes.items()):
                if node_id == node_identifier:
                    return idx
        
        return None
    
    def ingest_from_file(self, file_path):
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self.ingest(data, 'json')
        elif file_path.endswith('.csv'):
            return self.ingest(file_path, 'csv')
        else:
            return {'success': False, 'message': 'Unsupported file type'}
    
    def get_ingestion_summary(self):
        recent = list(self.ingestion_history)[-10:] if len(self.ingestion_history) > 10 else list(self.ingestion_history)
        
        format_counts = {}
        total_count = 0
        for entry in recent:
            format_counts[entry['format']] = format_counts.get(entry['format'], 0) + 1
            total_count += entry.get('count', 0)
        
        return {
            'total_ingestions': len(self.ingestion_history),
            'recent_ingestions': len(recent),
            'format_distribution': format_counts,
            'total_knowledge_added': total_count
        }