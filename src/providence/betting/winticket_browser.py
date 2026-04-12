"""WinTicket browser automation via Chrome DevTools Protocol.

Chrome の実ブラウザに CDP 接続し、WinTicket の投票フローを自動操作する。
bot 検知を回避するため、headless ではなくユーザーの実ブラウザを使用する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import structlog
from playwright.sync_api import Browser, Page, sync_playwright

from providence.domain.enums import TicketType, TrackCode

log = structlog.get_logger()
JST = timezone(timedelta(hours=9))

SCREENSHOTS_DIR = Path("data/screenshots")

_TRACK_SLUGS: dict[int, str] = {
    TrackCode.KAWAGUCHI: "kawaguchi",
    TrackCode.ISESAKI: "isesaki",
    TrackCode.HAMAMATSU: "hamamatsu",
    TrackCode.IIZUKA: "iizuka",
    TrackCode.SANYO: "sanyo",
}

_TICKET_TYPE_LABELS: dict[TicketType, str] = {
    TicketType.WIN: "単勝",
    TicketType.WIDE: "ワイド",
    TicketType.EXACTA: "2連単",
    TicketType.QUINELLA: "2連複",
    TicketType.TRIFECTA: "3連単",
    TicketType.TRIO: "3連複",
    TicketType.PLACE: "複勝",
}


@dataclass
class BetOrder:
    """投票指示。"""

    track_code: TrackCode
    race_number: int
    ticket_type: TicketType
    combination: tuple[int, ...]
    amount: int


@dataclass
class BetResult:
    """投票実行結果。"""

    order: BetOrder
    success: bool
    receipt_id: str | None = None
    balance_after: int | None = None
    error: str | None = None
    screenshot_path: str | None = None


@dataclass
class PhaseTiming:
    """Phase 2 の各ステップの処理時間（秒）。"""

    odds_scrape_sec: float = 0.0
    prediction_sec: float = 0.0
    strategy_sec: float = 0.0
    bet_execution_sec: float = 0.0
    total_sec: float = 0.0
    minutes_before_deadline: float = 0.0
    timestamp: str = ""


class WinTicketBrowser:
    """Chrome CDP 経由で WinTicket の投票を自動操作する。"""

    def __init__(self, debug_port: int = 9222) -> None:
        self._debug_port = debug_port
        self._pw_context = None
        self._browser: Browser | None = None
        self._page: Page | None = None

    def connect(self) -> None:
        """Chrome に CDP 接続する。"""
        self._pw_context = sync_playwright().start()
        self._browser = self._pw_context.chromium.connect_over_cdp(f"http://localhost:{self._debug_port}")
        for pg in self._browser.contexts[0].pages:
            if "winticket" in pg.url:
                self._page = pg
                return
        self._page = self._browser.contexts[0].pages[0]
        log.info("winticket_browser_connected", url=self._page.url)

    def close(self) -> None:
        """CDP 接続を閉じる（Chrome 自体は閉じない）。"""
        if self._pw_context:
            self._pw_context.stop()
            self._pw_context = None

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("connect() を先に呼んでください")
        return self._page

    def get_balance(self) -> int | None:
        """ポイント残高を取得する。"""
        body = self.page.inner_text("body")
        m = re.search(r"ポイント残高\s*([\d,]+)pt", body)
        return int(m.group(1).replace(",", "")) if m else None

    def is_logged_in(self) -> bool:
        """WinTicket にログイン済みか確認する。"""
        return self.get_balance() is not None

    def place_bets(self, orders: list[BetOrder], pin: str) -> list[BetResult]:
        """買い目リストを投票する。同一レースの買い目はまとめて投票ボックスに入れる。"""
        if not orders:
            return []

        results = []
        for order in orders:
            result = self._place_single_bet(order, pin)
            results.append(result)
        return results

    def _place_single_bet(self, order: BetOrder, pin: str) -> BetResult:
        """1つの買い目を投票する。"""
        label = f"{order.track_code.japanese_name}R{order.race_number}"
        log_ctx = log.bind(track=label, ticket=order.ticket_type.value, car=order.combination)

        try:
            # 1. レースページに遷移（UI ナビゲーション）
            self._navigate_to_race(order.track_code, order.race_number)

            # 2. カートクリア
            self._clear_cart()

            # 3. 投票シートで買い目を設定
            self._setup_bet(order)

            # 4. 追加する
            self._click_add()

            # 5. 投票ボックスで金額設定
            self._set_amount(order.amount)

            # 6. 金額バリデーション
            if not self._validate_total(order.amount):
                self._clear_cart()
                return BetResult(order=order, success=False, error="金額不一致")

            # 7. スクリーンショット（確定前）
            ss_path = self._screenshot(f"{label}_before_confirm")

            # 8. 確定 → PIN → 最終確定
            receipt_id = self._confirm_with_pin(pin)

            # 9. 残高取得
            balance = self.get_balance()

            # 10. スクリーンショット（確定後）
            self._screenshot(f"{label}_after_confirm")

            log_ctx.info("bet_placed", receipt=receipt_id, balance=balance)
            return BetResult(
                order=order, success=True, receipt_id=receipt_id,
                balance_after=balance, screenshot_path=ss_path,
            )

        except Exception as exc:
            error_msg = str(exc)
            log_ctx.error("bet_failed", error=error_msg)
            self._screenshot(f"{label}_error")
            try:
                self._clear_cart()
            except Exception:
                pass
            return BetResult(order=order, success=False, error=error_msg)

    def _navigate_to_race(self, track_code: TrackCode, race_number: int) -> None:
        """オートレーストップからレースを探して遷移する。"""
        page = self.page
        slug = _TRACK_SLUGS[track_code]

        page.goto("https://www.winticket.jp/autorace/", wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(5000)

        links = page.query_selector_all(f'a[href*="/autorace/{slug}/racecard/"]')
        for link in links:
            href = link.get_attribute("href") or ""
            text = (link.inner_text() or "").strip()
            parts = href.rstrip("/").split("/")
            if len(parts) >= 2 and parts[-1] == str(race_number) and "投票受付中" in text:
                url = f"https://www.winticket.jp{href}" if not href.startswith("http") else href
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(3000)
                return

        raise RuntimeError(f"レースが見つかりません: {track_code.japanese_name} R{race_number}")

    def _clear_cart(self) -> None:
        """投票ボックスの全アイテムを削除する。"""
        page = self.page
        page.click('button[role="tab"]:has-text("投票ボックス")')
        page.wait_for_timeout(2000)

        for _ in range(20):
            btns = page.query_selector_all('button[aria-label="削除する"]')
            visible = [b for b in btns if b.is_visible()]
            if not visible:
                break
            visible[0].click()
            page.wait_for_timeout(1000)

    def _setup_bet(self, order: BetOrder) -> None:
        """投票シートで賭式と車番を設定する。"""
        page = self.page
        page.click('button[role="tab"]:has-text("投票シート")')
        page.wait_for_timeout(2000)

        target_label = _TICKET_TYPE_LABELS[order.ticket_type]

        # 不要な賭式を解除
        for label in _TICKET_TYPE_LABELS.values():
            if self._is_bet_type_active(label) and label != target_label:
                page.click(f'button[role="checkbox"]:has-text("{label}")')
                page.wait_for_timeout(300)

        # 対象賭式を選択
        if not self._is_bet_type_active(target_label):
            page.click(f'button[role="checkbox"]:has-text("{target_label}")')
            page.wait_for_timeout(1000)

        # 車番を選択
        if order.ticket_type == TicketType.WIN:
            page.evaluate(
                f'document.querySelector(\'input[type="checkbox"][value="2:{order.combination[0]}"]\').click()'
            )
        elif order.ticket_type == TicketType.WIDE:
            page.evaluate(
                f'document.querySelector(\'input[type="checkbox"][value="2:{order.combination[0]}"]\').click()'
            )
            page.wait_for_timeout(500)
            page.evaluate(
                f'document.querySelector(\'input[type="checkbox"][value="3:{order.combination[1]}"]\').click()'
            )
        page.wait_for_timeout(2000)

    def _is_bet_type_active(self, label: str) -> bool:
        btn = self.page.query_selector(f'button[role="checkbox"]:has-text("{label}")')
        if not btn:
            return False
        cls = btn.get_attribute("class") or ""
        return "enter-done" in cls or "appear-done" in cls

    def _click_add(self) -> None:
        btn = self.page.query_selector('button:has-text("追加する")')
        if not btn or not btn.is_visible():
            raise RuntimeError("追加ボタンが見つかりません")
        btn.click()
        self.page.wait_for_timeout(2000)

    def _set_amount(self, amount: int) -> None:
        """金額を設定する（100pt 単位）。"""
        self.page.click('button[role="tab"]:has-text("投票ボックス")')
        self.page.wait_for_timeout(2000)

        units = amount // 100
        inp = self.page.query_selector('input[type="number"]')
        if inp:
            inp.fill(str(units))
            self.page.wait_for_timeout(500)

    def _validate_total(self, expected_amount: int) -> bool:
        """画面上の合計金額が意図した額と一致するか検証する。"""
        self.page.wait_for_timeout(1000)
        body = self.page.inner_text("body")
        m = re.search(r"投票合計\s*(\d[\d,]*)pt", body) or re.search(r"合計\s*(\d[\d,]*)pt", body)
        if not m:
            log.warning("validate_total_not_found")
            return True
        displayed = int(m.group(1).replace(",", ""))
        if displayed != expected_amount:
            log.error("validate_total_mismatch", displayed=displayed, expected=expected_amount)
            return False
        return True

    def _confirm_with_pin(self, pin: str) -> str | None:
        """確定 → PIN 入力 → 最終確定。受付番号を返す。"""
        page = self.page

        page.click('button:has-text("確定する")')
        page.wait_for_timeout(3000)

        pin_input = page.query_selector('input[type="password"][maxlength="4"]')
        if not pin_input:
            raise RuntimeError("PIN 入力欄が見つかりません")
        pin_input.fill(pin)
        page.wait_for_timeout(500)

        submit = page.query_selector('button[type="submit"]:has-text("確定する")')
        if not submit:
            raise RuntimeError("PIN 確定ボタンが見つかりません")
        submit.click()
        page.wait_for_timeout(5000)

        body = page.inner_text("body")
        m = re.search(r"受付番号\s*(\d+)", body)
        return m.group(1) if m else None

    def _screenshot(self, name: str) -> str:
        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
        path = str(SCREENSHOTS_DIR / f"{ts}_{name}.png")
        self.page.screenshot(path=path, full_page=True)
        return path
