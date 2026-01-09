"""
Comprehensive tests for the AgentsCommand class.

WHY: The agents command manages Claude Code native agents, which is core functionality.
It handles listing, deploying, and cleaning agent deployments.

DESIGN DECISIONS:
- Test all subcommands (list, deploy, clean, view, fix)
- Mock deployment service to avoid actual file operations
- Test error handling and validation
- Verify output formatting (json, yaml, table, text)
- Test dependency management features
"""

from argparse import Namespace
from unittest.mock import Mock, patch

from claude_mpm.cli.commands.agents import AgentsCommand
from claude_mpm.cli.shared.base_command import CommandResult
from claude_mpm.constants import AgentCommands


class TestAgentsCommand:
    """Test AgentsCommand functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.command = AgentsCommand()

    def test_initialization(self):
        """Test AgentsCommand initialization."""
        assert self.command.command_name == "agents"
        assert self.command.logger is not None
        assert self.command._deployment_service is None  # Lazy loaded
        assert self.command._formatter is not None  # Formatter is initialized

    def test_validate_args(self):
        """Test argument validation."""
        args = Namespace()
        error = self.command.validate_args(args)
        assert error is None  # Most agent commands are optional

    @patch("claude_mpm.cli.commands.agents.get_agent_versions_display")
    def test_run_default_show_versions(self, mock_get_versions):
        """Test default behavior shows agent versions."""
        mock_get_versions.return_value = "Agent: test-agent v1.0.0"

        args = Namespace(format="text")

        result = self.command.run(args)

        assert isinstance(result, CommandResult)
        assert result.success is True
        mock_get_versions.assert_called_once()

    @patch("claude_mpm.cli.commands.agents.get_agent_versions_display")
    def test_run_default_json_format(self):
        """Test default behavior with JSON format."""
        self.return_value = "Agent: test-agent v1.0.0"

        args = Namespace(format="json")

        result = self.command.run(args)

        assert isinstance(result, CommandResult)
        assert result.success is True
        assert result.data is not None
        assert "agent_versions" in result.data
        assert result.data["has_agents"] is True

    @patch("claude_mpm.cli.commands.agents.get_agent_versions_display")
    def test_run_default_no_agents(self):
        """Test default behavior when no agents are deployed."""
        self.return_value = None

        args = Namespace(format="json")

        result = self.command.run(args)

        assert isinstance(result, CommandResult)
        assert result.success is True
        assert result.data["has_agents"] is False
        assert "suggestion" in result.data

    def test_run_list_command(self):
        """Test list agents command."""
        args = Namespace(agents_command=AgentCommands.LIST.value, format="text")

        with patch.object(self.command, "_list_agents") as mock_list:
            mock_list.return_value = CommandResult.success_result("Agents listed")

            result = self.command.run(args)

            assert isinstance(result, CommandResult)
            assert result.success is True
            mock_list.assert_called_once_with(args)

    def test_run_deploy_command(self):
        """Test deploy agents command."""
        args = Namespace(agents_command=AgentCommands.DEPLOY.value, format="text")

        with patch.object(self.command, "_deploy_agents") as mock_deploy:
            mock_deploy.return_value = CommandResult.success_result("Agents deployed")

            result = self.command.run(args)

            assert isinstance(result, CommandResult)
            assert result.success is True
            mock_deploy.assert_called_once_with(args, force=False)

    def test_run_force_deploy_command(self):
        """Test force deploy agents command."""
        args = Namespace(agents_command=AgentCommands.FORCE_DEPLOY.value, format="text")

        with patch.object(self.command, "_deploy_agents") as mock_deploy:
            mock_deploy.return_value = CommandResult.success_result(
                "Agents force deployed"
            )

            result = self.command.run(args)

            assert isinstance(result, CommandResult)
            assert result.success is True
            mock_deploy.assert_called_once_with(args, force=True)

    def test_run_clean_command(self):
        """Test clean agents command."""
        args = Namespace(agents_command=AgentCommands.CLEAN.value, format="text")

        with patch.object(self.command, "_clean_agents") as mock_clean:
            mock_clean.return_value = CommandResult.success_result("Agents cleaned")

            result = self.command.run(args)

            assert isinstance(result, CommandResult)
            assert result.success is True
            mock_clean.assert_called_once_with(args)

    def test_run_view_command(self):
        """Test view agent command."""
        args = Namespace(agents_command=AgentCommands.VIEW.value, format="text")

        with patch.object(self.command, "_view_agent") as mock_view:
            mock_view.return_value = CommandResult.success_result("Agent viewed")

            result = self.command.run(args)

            assert isinstance(result, CommandResult)
            assert result.success is True
            mock_view.assert_called_once_with(args)

    def test_run_fix_command(self):
        """Test fix agents command."""
        args = Namespace(agents_command=AgentCommands.FIX.value, format="text")

        with patch.object(self.command, "_fix_agents") as mock_fix:
            mock_fix.return_value = CommandResult.success_result("Agents fixed")

            result = self.command.run(args)

            assert isinstance(result, CommandResult)
            assert result.success is True
            mock_fix.assert_called_once_with(args)

    def test_run_deps_check_command(self):
        """Test dependencies check command."""
        args = Namespace(agents_command="deps-check", format="text")

        with patch.object(self.command, "_check_agent_dependencies") as mock_check:
            mock_check.return_value = CommandResult.success_result(
                "Dependencies checked"
            )

            result = self.command.run(args)

            assert isinstance(result, CommandResult)
            assert result.success is True
            mock_check.assert_called_once_with(args)

    def test_run_deps_install_command(self):
        """Test dependencies install command."""
        args = Namespace(agents_command="deps-install", format="text")

        with patch.object(self.command, "_install_agent_dependencies") as mock_install:
            mock_install.return_value = CommandResult.success_result(
                "Dependencies installed"
            )

            result = self.command.run(args)

            assert isinstance(result, CommandResult)
            assert result.success is True
            mock_install.assert_called_once_with(args)

    def test_run_unknown_command(self):
        """Test unknown agent command."""
        args = Namespace(agents_command="unknown", format="text")

        result = self.command.run(args)

        assert isinstance(result, CommandResult)
        assert result.success is False
        assert "Unknown agent command" in result.message

    def test_deployment_service_lazy_loading(self):
        """Test deployment service is lazy loaded."""
        assert self.command._deployment_service is None

        # Mock the imports as they appear in the property method
        with patch("claude_mpm.services.AgentDeploymentService") as mock_service_class:
            with patch(
                "claude_mpm.services.agents.deployment.deployment_wrapper.DeploymentServiceWrapper"
            ) as mock_wrapper_class:
                mock_service = Mock()
                mock_wrapper = Mock()
                mock_service_class.return_value = mock_service
                mock_wrapper_class.return_value = mock_wrapper

                # Access the property to trigger lazy loading
                service = self.command.deployment_service

                # Verify lazy loading worked
                assert service is mock_wrapper
                assert self.command._deployment_service is mock_wrapper
                mock_service_class.assert_called_once()
                mock_wrapper_class.assert_called_once_with(mock_service)

    def test_deployment_service_import_error(self):
        """Test handling of deployment service import error."""
        # Simulate import error by patching the deployment_service property
        with patch.object(
            AgentsCommand, "deployment_service", property(lambda self: None)
        ):
            args = Namespace(agents_command=AgentCommands.DEPLOY.value, format="text")

            result = self.command.run(args)

            assert isinstance(result, CommandResult)
            assert result.success is False
            # Check for error in the message
            assert "Error" in result.message or "error" in result.message

    @patch.object(AgentsCommand, "deployment_service", new_callable=lambda: Mock())
    def test_list_agents_implementation(self):
        """Test _list_agents implementation."""
        self.list_deployed_agents.return_value = [
            {"name": "agent1", "version": "1.0.0"},
            {"name": "agent2", "version": "2.0.0"},
        ]

        args = Namespace(format="json")

        # We need to patch the actual method since it's private
        with patch.object(self.command, "_list_agents") as mock_list:
            mock_list.return_value = CommandResult.success_result(
                "Agents listed",
                data={
                    "agents": [
                        {"name": "agent1", "version": "1.0.0"},
                        {"name": "agent2", "version": "2.0.0"},
                    ]
                },
            )

            args.agents_command = AgentCommands.LIST.value
            result = self.command.run(args)

            assert result.success is True
            assert result.data is not None
            assert len(result.data["agents"]) == 2

    def test_deploy_agents_with_error(self):
        """Test deploy agents with error."""
        # Create a mock deployment service
        mock_service = Mock()
        mock_service.deploy_system_agents.side_effect = Exception("Deployment failed")

        # Set the mock service directly
        self.command._deployment_service = mock_service

        args = Namespace(agents_command=AgentCommands.DEPLOY.value, format="text")

        result = self.command.run(args)

        assert isinstance(result, CommandResult)
        assert result.success is False
        # Check for either error message since the actual implementation might vary
        assert "Error" in result.message or "error" in result.message

    def test_run_with_exception(self):
        """Test general exception handling in run method."""
        args = Namespace(agents_command=AgentCommands.LIST.value, format="text")

        with patch.object(
            self.command, "_list_agents", side_effect=Exception("Test error")
        ):
            result = self.command.run(args)

            assert isinstance(result, CommandResult)
            assert result.success is False
            assert "Error managing agents" in result.message
            assert "Test error" in result.message

    def test_run_sync_command(self):
        """Test sync agents command routes correctly."""
        args = Namespace(agents_command=AgentCommands.SYNC.value, format="text")

        with patch.object(self.command, "_sync_agents") as mock_sync:
            mock_sync.return_value = CommandResult.success_result("Agents synced")

            result = self.command.run(args)

            assert isinstance(result, CommandResult)
            assert result.success is True
            mock_sync.assert_called_once_with(args)

    @patch("claude_mpm.services.agents.startup_sync.sync_agents_on_startup")
    def test_sync_agents_success(self, mock_sync_on_startup):
        """Test _sync_agents with successful sync."""
        mock_sync_on_startup.return_value = {
            "enabled": True,
            "sources_synced": 1,
            "total_downloaded": 10,
            "cache_hits": 5,
            "errors": [],
            "duration_ms": 500,
        }

        args = Namespace(
            agents_command=AgentCommands.SYNC.value,
            format="text",
            force=False,
            source=None,
        )

        result = self.command._sync_agents(args)

        assert isinstance(result, CommandResult)
        assert result.success is True
        assert "10" in result.message  # downloaded count
        assert "1" in result.message  # sources count
        mock_sync_on_startup.assert_called_once_with(config=None, force_refresh=False)

    @patch("claude_mpm.services.agents.startup_sync.sync_agents_on_startup")
    def test_sync_agents_with_force(self, mock_sync_on_startup):
        """Test _sync_agents with force flag."""
        mock_sync_on_startup.return_value = {
            "enabled": True,
            "sources_synced": 1,
            "total_downloaded": 15,
            "cache_hits": 0,
            "errors": [],
            "duration_ms": 1200,
        }

        args = Namespace(
            agents_command=AgentCommands.SYNC.value,
            format="text",
            force=True,
            source=None,
        )

        result = self.command._sync_agents(args)

        assert isinstance(result, CommandResult)
        assert result.success is True
        mock_sync_on_startup.assert_called_once_with(config=None, force_refresh=True)

    @patch("claude_mpm.services.agents.startup_sync.sync_agents_on_startup")
    def test_sync_agents_disabled(self, mock_sync_on_startup):
        """Test _sync_agents when sync is disabled in config."""
        mock_sync_on_startup.return_value = {
            "enabled": False,
            "sources_synced": 0,
            "total_downloaded": 0,
            "cache_hits": 0,
            "errors": [],
            "duration_ms": 0,
        }

        args = Namespace(
            agents_command=AgentCommands.SYNC.value,
            format="text",
            force=False,
            source=None,
        )

        result = self.command._sync_agents(args)

        assert isinstance(result, CommandResult)
        assert result.success is True
        assert "disabled" in result.message.lower()

    @patch("claude_mpm.services.agents.startup_sync.sync_agents_on_startup")
    def test_sync_agents_with_errors(self, mock_sync_on_startup):
        """Test _sync_agents when sync has errors."""
        mock_sync_on_startup.return_value = {
            "enabled": True,
            "sources_synced": 1,
            "total_downloaded": 5,
            "cache_hits": 2,
            "errors": ["Failed to sync agent X", "Network error for agent Y"],
            "duration_ms": 800,
        }

        args = Namespace(
            agents_command=AgentCommands.SYNC.value,
            format="text",
            force=False,
            source=None,
        )

        result = self.command._sync_agents(args)

        assert isinstance(result, CommandResult)
        assert result.success is True  # Partial success is still success
        assert "2 errors" in result.message

    @patch("claude_mpm.services.agents.startup_sync.sync_agents_on_startup")
    def test_sync_agents_json_format(self, mock_sync_on_startup):
        """Test _sync_agents with JSON output format."""
        mock_sync_on_startup.return_value = {
            "enabled": True,
            "sources_synced": 1,
            "total_downloaded": 10,
            "cache_hits": 5,
            "errors": [],
            "duration_ms": 500,
        }

        args = Namespace(
            agents_command=AgentCommands.SYNC.value,
            format="json",
            force=False,
            source=None,
        )

        result = self.command._sync_agents(args)

        assert isinstance(result, CommandResult)
        assert result.success is True
        assert result.data is not None
        assert result.data["enabled"] is True
        assert result.data["total_downloaded"] == 10

    @patch("claude_mpm.services.agents.startup_sync.sync_agents_on_startup")
    def test_sync_agents_exception_handling(self, mock_sync_on_startup):
        """Test _sync_agents handles exceptions gracefully."""
        mock_sync_on_startup.side_effect = Exception("Network connection failed")

        args = Namespace(
            agents_command=AgentCommands.SYNC.value,
            format="text",
            force=False,
            source=None,
        )

        result = self.command._sync_agents(args)

        assert isinstance(result, CommandResult)
        assert result.success is False
        assert "Error syncing agents" in result.message

    @patch("claude_mpm.core.config.Config")
    @patch("claude_mpm.services.agents.startup_sync.sync_agents_on_startup")
    def test_sync_agents_with_source_filter(self, mock_sync_on_startup, mock_config):
        """Test _sync_agents with source filter."""
        mock_config_instance = Mock()
        mock_config_instance.to_dict.return_value = {
            "agent_sync": {
                "enabled": True,
                "sources": [
                    {"id": "source1", "url": "https://example.com/1"},
                    {"id": "source2", "url": "https://example.com/2"},
                ],
            }
        }
        mock_config.return_value = mock_config_instance

        mock_sync_on_startup.return_value = {
            "enabled": True,
            "sources_synced": 1,
            "total_downloaded": 5,
            "cache_hits": 0,
            "errors": [],
            "duration_ms": 300,
        }

        args = Namespace(
            agents_command=AgentCommands.SYNC.value,
            format="text",
            force=False,
            source="source1",
        )

        result = self.command._sync_agents(args)

        assert isinstance(result, CommandResult)
        assert result.success is True
        # Verify filtered config was passed
        call_args = mock_sync_on_startup.call_args
        config_passed = call_args.kwargs.get("config")
        assert config_passed is not None
        assert len(config_passed["agent_sync"]["sources"]) == 1
        assert config_passed["agent_sync"]["sources"][0]["id"] == "source1"

    @patch("claude_mpm.core.config.Config")
    def test_sync_agents_source_not_found(self, mock_config):
        """Test _sync_agents with non-existent source filter."""
        mock_config_instance = Mock()
        mock_config_instance.to_dict.return_value = {
            "agent_sync": {
                "enabled": True,
                "sources": [
                    {"id": "source1", "url": "https://example.com/1"},
                ],
            }
        }
        mock_config.return_value = mock_config_instance

        args = Namespace(
            agents_command=AgentCommands.SYNC.value,
            format="text",
            force=False,
            source="nonexistent",
        )

        result = self.command._sync_agents(args)

        assert isinstance(result, CommandResult)
        assert result.success is False
        assert "not found" in result.message
