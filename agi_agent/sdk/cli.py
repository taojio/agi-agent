import argparse
import json
import sys
import os
from typing import Dict, Any
from .api_client import AGIAgentAPIClient
from ..plugins.plugin_template.template import write_plugin_file


class AGIAgentCLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog="agi-agent",
            description="AGI Agent Command Line Interface",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        self.subparsers = self.parser.add_subparsers(dest="command", help="Available commands")
        self._setup_subcommands()
    
    def _setup_subcommands(self):
        health_parser = self.subparsers.add_parser("health", help="Check agent health status")
        
        status_parser = self.subparsers.add_parser("status", help="Get agent status")
        
        memory_parser = self.subparsers.add_parser("memory", help="Memory management")
        memory_subparsers = memory_parser.add_subparsers(dest="memory_command")
        
        mem_list_parser = memory_subparsers.add_parser("list", help="List memory tiers")
        mem_search_parser = memory_subparsers.add_parser("search", help="Search memory")
        mem_search_parser.add_argument("query", help="Search query")
        mem_search_parser.add_argument("--tier", help="Memory tier")
        mem_search_parser.add_argument("--limit", type=int, default=10)
        mem_add_parser = memory_subparsers.add_parser("add", help="Add memory")
        mem_add_parser.add_argument("content", help="Memory content")
        mem_add_parser.add_argument("--tier", default="short_term")
        mem_add_parser.add_argument("--category", default="default")
        
        plugin_parser = self.subparsers.add_parser("plugin", help="Plugin management")
        plugin_subparsers = plugin_parser.add_subparsers(dest="plugin_command")
        
        plugin_list_parser = plugin_subparsers.add_parser("list", help="List plugins")
        plugin_load_parser = plugin_subparsers.add_parser("load", help="Load plugin")
        plugin_load_parser.add_argument("name", help="Plugin name")
        plugin_activate_parser = plugin_subparsers.add_parser("activate", help="Activate plugin")
        plugin_activate_parser.add_argument("name", help="Plugin name")
        plugin_create_parser = plugin_subparsers.add_parser("create", help="Create plugin template")
        plugin_create_parser.add_argument("name", help="Plugin name")
        plugin_create_parser.add_argument("--description", default="My Plugin")
        plugin_create_parser.add_argument("--type", default="utility")
        plugin_create_parser.add_argument("--output", help="Output directory")
        
        service_parser = self.subparsers.add_parser("service", help="Service management")
        service_subparsers = service_parser.add_subparsers(dest="service_command")
        
        service_list_parser = service_subparsers.add_parser("list", help="List registered services")
        service_status_parser = service_subparsers.add_parser("status", help="Get service registry status")
        
        gateway_parser = self.subparsers.add_parser("gateway", help="Gateway management")
        gateway_subparsers = gateway_parser.add_subparsers(dest="gateway_command")
        
        gateway_routes_parser = gateway_subparsers.add_parser("routes", help="List gateway routes")
        gateway_status_parser = gateway_subparsers.add_parser("status", help="Get gateway status")
        
        chat_parser = self.subparsers.add_parser("chat", help="Send message to agent")
        chat_parser.add_argument("message", help="Message to send")
        chat_parser.add_argument("--context", default="")
    
    def run(self, args=None):
        args = self.parser.parse_args(args)
        
        if not args.command:
            self.parser.print_help()
            return
        
        client = AGIAgentAPIClient()
        
        try:
            if args.command == "health":
                result = client.health_check()
                self._print_result(result)
            
            elif args.command == "status":
                result = client.get_agent_status()
                self._print_result(result)
            
            elif args.command == "memory":
                self._handle_memory_command(args, client)
            
            elif args.command == "plugin":
                self._handle_plugin_command(args, client)
            
            elif args.command == "service":
                self._handle_service_command(args, client)
            
            elif args.command == "gateway":
                self._handle_gateway_command(args, client)
            
            elif args.command == "chat":
                result = client.send_message(args.message, args.context)
                self._print_result(result)
        
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    def _handle_memory_command(self, args, client):
        if args.memory_command == "list":
            result = client.get_memory_tiers()
        elif args.memory_command == "search":
            result = client.search_memory(args.query, args.tier, args.limit)
        elif args.memory_command == "add":
            result = client.add_memory(args.content, args.tier, args.category)
        else:
            print("Unknown memory command", file=sys.stderr)
            return
        self._print_result(result)
    
    def _handle_plugin_command(self, args, client):
        if args.plugin_command == "list":
            result = client.get_plugins()
        elif args.plugin_command == "load":
            result = client.load_plugin(args.name)
        elif args.plugin_command == "activate":
            result = client.activate_plugin(args.name)
        elif args.plugin_command == "create":
            filepath = write_plugin_file(
                args.name, 
                output_dir=args.output,
                plugin_description=args.description,
                plugin_type=args.type
            )
            print(f"Plugin template created at: {filepath}")
            return
        else:
            print("Unknown plugin command", file=sys.stderr)
            return
        self._print_result(result)
    
    def _handle_service_command(self, args, client):
        if args.service_command == "list":
            result = client.get_service_registry()
        elif args.service_command == "status":
            result = client.get_service_registry()
        else:
            print("Unknown service command", file=sys.stderr)
            return
        self._print_result(result)
    
    def _handle_gateway_command(self, args, client):
        if args.gateway_command == "routes":
            result = client.get_gateway_routes()
        elif args.gateway_command == "status":
            result = client.get_gateway_routes()
        else:
            print("Unknown gateway command", file=sys.stderr)
            return
        self._print_result(result)
    
    def _print_result(self, result: Dict[str, Any]):
        print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    cli = AGIAgentCLI()
    cli.run()


if __name__ == "__main__":
    main()