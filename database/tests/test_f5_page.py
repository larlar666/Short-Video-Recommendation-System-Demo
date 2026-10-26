import importlib.util
import unittest


def render_f5():
    import pandas as pd
    import streamlit_app as app
    from unittest.mock import patch

    vid = "7477203555078786350"
    videos = pd.DataFrame({"id": [vid], "desc": ["test"], "play_count": [80], "fields_list": [["Music"]]})
    # JSONB key order can differ from the original CSV.
    metrics = {f"round_{rnd}": [float(rnd)] for rnd in range(30, 0, -1)}
    series = pd.DataFrame({"video_id": [vid], "pred_total_play": [459.0], **metrics})
    with patch.object(app, "cached_video_watch_counts", return_value=pd.Series({1: 2, 3: 1, 30: 3})):
        app.f5_panel(videos, series, None)


def render_f5_timeout():
    import pandas as pd
    import streamlit_app as app
    from unittest.mock import patch

    vid = "7477203555078786350"
    videos = pd.DataFrame({"id": [vid], "desc": ["test"], "play_count": [80], "fields_list": [["Music"]]})
    series = pd.DataFrame({"video_id": [vid], **{f"round_{rnd}": [1.] for rnd in range(1, 31)}})
    error = type("ConnectionTimeout", (Exception,), {})("postgresql://secret-credentials")
    with patch.object(app, "cached_video_watch_counts", side_effect=error), patch.object(app.logger, "error"):
        app.f5_panel(videos, series, None)


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "需要安装界面依赖")
class F5PageTests(unittest.TestCase):
    def test_cloud_prediction_shape_renders_metrics_and_chart(self):
        from streamlit.testing.v1 import AppTest
        page = AppTest.from_function(render_f5).run(timeout=15)
        self.assertFalse(page.exception)
        self.assertFalse(page.error)
        self.assertEqual(page.metric[2].value, "6")
        self.assertEqual(len(page.get("vega_lite_chart")) + len(page.get("arrow_vega_lite_chart")), 1)

    def test_timeout_explains_network_without_leaking_credentials(self):
        from streamlit.testing.v1 import AppTest
        page = AppTest.from_function(render_f5_timeout).run(timeout=15)
        self.assertFalse(page.exception)
        self.assertIn("连接超时", page.error[0].value)
        self.assertNotIn("secret-credentials", str(page))
        self.assertEqual(page.metric[1].value, "27")
        self.assertEqual(page.metric[2].value, "暂不可用")
        self.assertEqual(len(page.get("vega_lite_chart")) + len(page.get("arrow_vega_lite_chart")), 1)


if __name__ == "__main__":
    unittest.main()
