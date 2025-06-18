import asyncio
from asyncua import Client, ua

class OpcuaClientConnector:
    """
    一个用于连接到本地OPC UA服务器并发送警报的类。
    """
    # --- 配置 ---
    # 这个URL必须和你server.py里设置的完全一样！
    SERVER_URL = "opc.tcp://0.0.0.0:4840/SafetyServer/server/"
    # 这个URI也必须和server.py里注册的namespace一样
    NAMESPACE_URI = "http://solutions.com/workersafety-server"
    
    # 定义警报类型和对应的数字代码
    ALERT_CODES = {
        "fall": 1,
        "no_helmet": 2,
        "stationary": 3
        # "smoking": 4
        # 可以继续添加更多，比如 "trespass": 5
    }
    
    # 映射：哪个摄像头，哪个警报类型，对应服务器上的哪个变量名
    # 格式: (camera_id, alert_type) -> "VariableName"
    # camera_id可以是RTSP流IP的最后一部分，或者视频文件名等标识
    # 'default' 是一个备用变量，如果没有为特定摄像头配置
    VARIABLE_MAP = {
        ('default', 'fall'): "Cam0 Unsafe Exit", 
        ('default', 'no_helmet'): "Cam0 Trespass",
        ('default', 'stationary'): "Cam58 Overtime Parking",
        # ('default', 'smoking'): "Cam71 Trespass", # 假设用这个变量来代表吸烟

        # 可以为特定的摄像头配置特定的变量
        # 比如，如果你的视频源是 "172.30.40.58"
        # ('58', 'fall'): "Cam58 Unsafe Exit",
        # ('58', 'no_helmet'): "Cam58 Trespass",
    }

    def __init__(self):
        self.server_url = self.SERVER_URL
        self.namespace_uri = self.NAMESPACE_URI
        self.namespace_idx = None
        self.client = None
        print("OPC UA Connector Initialized.")

    async def connect(self):
        """连接到OPC UA服务器"""
        try:
            # 使用 with 语句可以确保断开连接
            async with Client(url=self.server_url) as client:
                self.namespace_idx = await client.get_namespace_index(self.namespace_uri)
                print(f"Connected to OPC UA Server. Namespace '{self.namespace_uri}' has index {self.namespace_idx}")
                # 在实际应用中，你可能想保持连接，而不是每次都重连。
                # 但对于简单发送，每次重连更稳定。
                # 这里我们先示范一个完整的连接->操作->断开流程。
                # 这个connect方法在这里主要用于测试。
                return True
        except Exception as e:
            print(f"[ERROR] OPC UA Connection failed: {e}")
            return False

    def get_variable_name(self, camera_id, alert_type):
        """根据摄像头ID和警报类型获取OPC UA变量名"""
        # 尝试精确匹配
        key = (str(camera_id), alert_type)
        if key in self.VARIABLE_MAP:
            return self.VARIABLE_MAP[key]
        
        # 如果没有精确匹配，尝试使用默认值
        default_key = ('default', alert_type)
        if default_key in self.VARIABLE_MAP:
            print(f"[OPC UA] No specific variable for Cam '{camera_id}' / '{alert_type}', using default '{self.VARIABLE_MAP[default_key]}'")
            return self.VARIABLE_MAP[default_key]
            
        return None


    async def send_alert_internal(self, camera_id, alert_type):
        """内部异步函数，负责连接、写入和断开"""
        alert_code = self.ALERT_CODES.get(alert_type)
        if alert_code is None:
            print(f"[WARNING] Unknown alert type for OPC UA: {alert_type}")
            return

        variable_name = self.get_variable_name(camera_id, alert_type)
        if variable_name is None:
            print(f"[WARNING] No OPC UA variable mapped for alert type: {alert_type}")
            return

        try:
            async with Client(url=self.server_url) as client:
                ns_idx = await client.get_namespace_index(self.namespace_uri)
                
                # 构建要操作的节点的路径
                # 路径是：Objects -> Cameras -> YourVariableName
                var_node_path = f"0:Objects/{ns_idx}:Cameras/{ns_idx}:{variable_name}"
                var_node = await client.nodes.root.get_child(var_node_path.split("/"))
                
                # 写入值
                print(f"[OPC UA] Sending alert. Writing '{alert_code}' to node '{variable_name}' ({var_node})")
                await var_node.write_value(ua.Variant(alert_code, ua.VariantType.Int32)) # 假设变量是32位整数
                print(f"[OPC UA] Successfully wrote value.")

        except Exception as e:
            print(f"[ERROR] Failed to send OPC UA alert: {e}")

    def send_alert(self, camera_id, alert_type):
        """
        同步的外部接口。它会启动一个异步事件循环来发送警报。
        这样，我们的主检测线程就不会被阻塞。
        """
        print(f"Queueing OPC UA alert: Cam '{camera_id}', Type '{alert_type}'")
        # 使用asyncio.run在新线程中运行异步代码，避免阻塞主线程
        # 更健壮的做法是使用一个全局的事件循环和队列，但这个方法对于发送少量警报是有效的
        try:
            asyncio.run(self.send_alert_internal(camera_id, alert_type))
        except Exception as e:
            # 如果已有事件循环在运行，asyncio.run会报错。此时可以用create_task
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.send_alert_internal(camera_id, alert_type))
            except RuntimeError:
                print(f"[ERROR] Could not find or create asyncio event loop to send OPC UA alert. Error: {e}")