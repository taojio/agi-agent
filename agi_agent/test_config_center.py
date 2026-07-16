import os
import sys
import tempfile
import json
import logging

logging.basicConfig(level=logging.INFO)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

exec(open(os.path.join(os.path.dirname(__file__), "config_runtime/config_center.py")).read())


def test_config_center():
    print("=" * 60)
    print("T005: 配置中心化管理 - 功能测试")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        cm = ConfigManager(config_dir=temp_dir)

        print("\n1. 测试配置加载优先级")
        cm.load_defaults({"app.name": "DefaultApp", "app.port": 8080})
        cm.set("app.name", "FileApp", ConfigSource.FILE)
        cm.set("app.name", "EnvApp", ConfigSource.ENVIRONMENT)
        result = cm.get("app.name")
        print(f"   优先级测试结果: {result} (预期: EnvApp)")
        assert result == "EnvApp", f"优先级测试失败: {result}"

        print("\n2. 测试模块级配置")
        cm.load_defaults({"timeout": 30, "retry_count": 3}, module="network")
        cm.load_defaults({"timeout": 60}, module="database")
        network_config = cm.get_module_config("network")
        database_config = cm.get_module_config("database")
        print(f"   network.timeout: {network_config.get('network.timeout')}")
        print(f"   database.timeout: {database_config.get('database.timeout')}")
        assert cm.get("timeout", module="network") == 30
        assert cm.get("timeout", module="database") == 60

        print("\n3. 测试敏感配置加密")
        cm.set("database.password", "secret123", sensitive=True)
        encrypted_value = cm._configs["database.password"].value
        decrypted_value = cm.get("database.password")
        print(f"   加密后: {encrypted_value[:20]}...")
        print(f"   解密后: {decrypted_value}")
        assert decrypted_value == "secret123"

        print("\n4. 测试配置监听器")
        changes = []

        def listener(change):
            changes.append(f"{change.key}: {change.old_value} -> {change.new_value}")

        cm.register_listener("app.*", listener)
        cm.set("app.port", 9090)
        cm.set("app.debug", True)
        print(f"   监听器捕获变更: {changes}")
        assert len(changes) == 2

        print("\n5. 测试配置变更历史")
        history = cm.get_change_history()
        print(f"   变更历史记录数: {len(history)}")
        assert len(history) >= 2

        print("\n6. 测试配置统计")
        stats = cm.get_stats()
        print(f"   统计信息: {json.dumps(stats, indent=2)}")
        assert stats["total_configs"] > 0

        print("\n7. 测试配置快照与恢复")
        version_id = cm.snapshot()
        print(f"   创建快照: {version_id}")
        cm.set("app.name", "ModifiedApp")
        snapshots = cm.list_snapshots()
        assert len(snapshots) > 0
        restored = cm.restore_snapshot(version_id)
        assert restored
        assert cm.get("app.name") == "EnvApp"

        print("\n8. 测试配置模式验证")
        cm.register_schema("app.port", {"type": "int", "min": 1, "max": 65535})
        cm.set("app.port", "invalid")
        print("   模式验证测试完成")

        print("\n9. 测试配置链式加载")
        defaults_path = os.path.join(temp_dir, "defaults.json")
        config_path = os.path.join(temp_dir, "config.json")
        local_path = os.path.join(temp_dir, "local.json")

        with open(defaults_path, "w") as f:
            json.dump({"chain.test": "default"}, f)
        with open(config_path, "w") as f:
            json.dump({"chain.test": "file"}, f)
        with open(local_path, "w") as f:
            json.dump({"chain.test": "local"}, f)

        cm2 = ConfigManager(config_dir=temp_dir)
        cm2.load_config_chain()
        result = cm2.get("chain.test")
        print(f"   链式加载结果: {result} (预期: local)")
        assert result == "local"

        print("\n" + "=" * 60)
        print("所有测试通过!")
        print("=" * 60)


if __name__ == "__main__":
    test_config_center()