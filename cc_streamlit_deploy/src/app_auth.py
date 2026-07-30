"""Shared-password authentication for the Streamlit application."""

from __future__ import annotations

import hmac
from enum import Enum
from typing import Any, MutableMapping, Optional

from src.api_keys import get_app_password


AUTH_SESSION_KEY = "authenticated"
PASSWORD_WIDGET_KEY = "_login_password"
MISSING_PASSWORD_MESSAGE = "未配置 APP_PASSWORD，请在 Streamlit Secrets 中设置。"


class AuthenticationResult(str, Enum):
    MISSING_CONFIGURATION = "missing_configuration"
    SUCCESS = "success"
    FAILURE = "failure"


def verify_password(candidate: str, configured_password: str) -> bool:
    if not configured_password:
        return False
    return hmac.compare_digest(
        str(candidate).encode("utf-8"),
        str(configured_password).encode("utf-8"),
    )


def authenticate(
    session_state: MutableMapping[str, Any],
    candidate: str,
    configured_password: str,
) -> AuthenticationResult:
    if not configured_password:
        session_state[AUTH_SESSION_KEY] = False
        session_state.pop(PASSWORD_WIDGET_KEY, None)
        return AuthenticationResult.MISSING_CONFIGURATION
    authenticated = verify_password(candidate, configured_password)
    session_state[AUTH_SESSION_KEY] = authenticated
    session_state.pop(PASSWORD_WIDGET_KEY, None)
    return AuthenticationResult.SUCCESS if authenticated else AuthenticationResult.FAILURE


def logout(session_state: MutableMapping[str, Any]) -> None:
    session_state.pop(AUTH_SESSION_KEY, None)
    session_state.pop(PASSWORD_WIDGET_KEY, None)


def is_authenticated(session_state: MutableMapping[str, Any]) -> bool:
    return session_state.get(AUTH_SESSION_KEY) is True


def require_authentication(st_module: Any, configured_password: Optional[str] = None) -> bool:
    password = get_app_password() if configured_password is None else configured_password
    if is_authenticated(st_module.session_state) and password:
        return True

    st_module.markdown(
        "<h1 style='text-align:center;'>TitaniumFatigueChat</h1>",
        unsafe_allow_html=True,
    )
    st_module.markdown(
        "<p style='text-align:center;color:#666;'>请输入共享访问密码以进入科研助手。</p>",
        unsafe_allow_html=True,
    )
    if not password:
        st_module.error(MISSING_PASSWORD_MESSAGE)
        return False

    candidate = st_module.text_input(
        "访问密码",
        type="password",
        key=PASSWORD_WIDGET_KEY,
        placeholder="请输入访问密码",
    )
    if st_module.button("登录", type="primary", use_container_width=True):
        result = authenticate(st_module.session_state, candidate, password)
        if result is AuthenticationResult.SUCCESS:
            st_module.rerun()
        else:
            st_module.error("访问密码错误")
    return False


def render_logout_control(st_module: Any) -> None:
    if st_module.button("退出登录", use_container_width=True, key="logout_button"):
        logout(st_module.session_state)
        st_module.rerun()
