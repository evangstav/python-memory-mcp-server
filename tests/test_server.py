import pytest
from memory_mcp_server.main import create_server
from mcp.server import Server
import mcp.types as types

@pytest.mark.asyncio
async def test_server_initialization(temp_db_path):
    """Test server initialization and configuration."""
    server = await create_server(db_path=temp_db_path)
    assert isinstance(server, Server)
    assert server.name == "memory-mcp-server"

@pytest.mark.asyncio
async def test_server_list_tools(temp_db_path):
    """Test the server's list_tools endpoint."""
    server = await create_server(db_path=temp_db_path)
    
    # Get the list_tools handler
    tools_handler = server._request_handlers.get("tools/list")
    assert tools_handler is not None
    
    # Call the handler
    result = await tools_handler({})
    assert isinstance(result, dict)
    assert "tools" in result
    
    # Verify expected tools are present
    tool_names = {tool.name for tool in result["tools"]}
    expected_tools = {
        "create_entity",
        "create_relation",
        "add_observations",
        "delete_entity",
        "delete_relation",
        "search_nodes",
        "read_graph"
    }
    assert expected_tools.issubset(tool_names)

@pytest.mark.asyncio
async def test_server_call_tool(temp_db_path):
    """Test the server's call_tool endpoint."""
    server = await create_server(db_path=temp_db_path)
    
    # Get the call_tool handler
    call_handler = server._request_handlers.get("tools/call")
    assert call_handler is not None
    
    # Test creating an entity
    create_params = {
        "name": "create_entity",
        "arguments": {
            "name": "TestEntity",
            "entityType": "TestType",
            "observations": ["Test observation"]
        }
    }
    result = await call_handler(types.CallToolRequest(**create_params))
    assert isinstance(result, types.CallToolResult)
    assert not result.isError

    # Test searching for the created entity
    search_params = {
        "name": "search_nodes",
        "arguments": {
            "query": "TestEntity"
        }
    }
    result = await call_handler(types.CallToolRequest(**search_params))
    assert isinstance(result, types.CallToolResult)
    assert not result.isError
    # Parse the result content
    content = result.content[0]
    assert isinstance(content, types.TextContent)
    assert "TestEntity" in content.text