"""
配置管理
统一从项目根目录的 .env 文件加载配置
"""

import os
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
# 路径: MiroFish/.env (相对于 backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # 如果根目录没有 .env，尝试加载环境变量（用于生产环境）
    load_dotenv(override=True)


class Config:
    """Flask配置类"""
    
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # JSON配置 - 禁用ASCII转义，让中文直接显示（而不是 \uXXXX 格式）
    JSON_AS_ASCII = False
    
    # LLM配置（统一使用OpenAI格式）
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')
    
    # Zep配置（仅旧版图谱流程需要；fundsim 模块不依赖 Zep）
    ZEP_API_KEY = os.environ.get('ZEP_API_KEY')

    # --- fundsim: simulation d'allocation LP -> fonds privés ---
    # Modèle économique pour la boucle mensuelle, modèle fort pour l'analyse finale.
    FUNDSIM_MODEL_FAST = os.environ.get('FUNDSIM_MODEL_FAST') or os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')
    FUNDSIM_MODEL_STRONG = os.environ.get('FUNDSIM_MODEL_STRONG') or os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')
    # Garde-fou de coût : nombre maximal d'appels LLM par run.
    FUNDSIM_MAX_CALLS_PER_RUN = int(os.environ.get('FUNDSIM_MAX_CALLS_PER_RUN', '500'))
    # Dry-run : aucun appel réseau, réponses factices (tests et démos hors ligne).
    FUNDSIM_DRY_RUN = os.environ.get('FUNDSIM_DRY_RUN', 'false').lower() == 'true'
    FUNDSIM_DATA_DIR = os.environ.get(
        'FUNDSIM_DATA_DIR', os.path.join(os.path.dirname(__file__), '../uploads/fundsim')
    )
    
    # 文件上传配置
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}
    
    # 文本处理配置
    DEFAULT_CHUNK_SIZE = 500  # 默认切块大小
    DEFAULT_CHUNK_OVERLAP = 50  # 默认重叠大小
    
    # OASIS模拟配置
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')
    
    # OASIS平台可用动作配置
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]
    
    # Report Agent配置
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))
    
    @classmethod
    def validate(cls) -> list[str]:
        """验证必要配置"""
        errors: list[str] = []
        if not cls.LLM_API_KEY and not cls.FUNDSIM_DRY_RUN:
            errors.append("LLM_API_KEY 未配置")
        # ZEP_API_KEY n'est plus requis au démarrage : les endpoints du flux Zep
        # d'origine le vérifient eux-mêmes, et fundsim n'utilise pas Zep.
        return errors

