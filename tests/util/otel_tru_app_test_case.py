from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Type

from trulens.core.app import App
from trulens.core.database.connector import DefaultDBConnector
from trulens.core.session import TruSession

from tests.util.otel_test_case import OtelTestCase


@dataclass
class TestAppInfo:
    app: Any
    main_method: Callable
    TruAppClass: Type[App]


class OtelTruAppTestCase(OtelTestCase):
    @staticmethod
    @abstractmethod
    def _create_test_app_info() -> TestAppInfo:
        pass

    def test_with_existing_tru_session(self) -> None:
        self.clear_TruSession_singleton()
        test_app_info = self._create_test_app_info()
        connector1 = DefaultDBConnector()
        connector2 = DefaultDBConnector()
        self.assertIsNot(connector1, connector2)
        TruSession(connector=connector1)
        with self.assertRaisesRegex(
            ValueError,
            "Already created `TruSession` with different `connector`!",
        ):
            test_app_info.TruAppClass(
                test_app_info.app,
                main_method=test_app_info.main_method,
                connector=connector2,
            )
        tru_app = test_app_info.TruAppClass(
            test_app_info.app,
            main_method=test_app_info.main_method,
            connector=connector1,
        )
        self.assertIs(connector1, tru_app.connector)

    def test_with_new_tru_session(self) -> None:
        self.clear_TruSession_singleton()
        test_app_info = self._create_test_app_info()
        connector = DefaultDBConnector()
        tru_app = test_app_info.TruAppClass(
            test_app_info.app,
            main_method=test_app_info.main_method,
            connector=connector,
        )
        self.assertIs(connector, tru_app.connector)
        self.assertIs(connector, TruSession().connector)
