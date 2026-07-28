"""Pytest fixtures: Game process wrapper for unit tests."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import pytest
from sts2_bench.seeds import DEFAULT_TEST_SEED

BENCH_ROOT = Path(__file__).resolve().parents[1]
STS2_CLI_ROOT = Path(
    os.environ.get("STS2_CLI_ROOT", str(BENCH_ROOT.parent / "sts2-cli"))
).resolve()
LOCAL_DOTNET_DIR = STS2_CLI_ROOT / ".tools" / "dotnet"
LOCAL_DOTNET = LOCAL_DOTNET_DIR / ("dotnet.exe" if os.name == "nt" else "dotnet")
MAC_ARM_DOTNET = Path(os.path.expanduser("~/.dotnet-arm64/dotnet"))
if LOCAL_DOTNET.is_file():
    DOTNET = str(LOCAL_DOTNET)
elif MAC_ARM_DOTNET.is_file():
    DOTNET = str(MAC_ARM_DOTNET)
else:
    DOTNET = shutil.which("dotnet") or str(MAC_ARM_DOTNET)
PROJECT = str(STS2_CLI_ROOT / "src" / "Sts2Headless" / "Sts2Headless.csproj")
HEADLESS_DLL = STS2_CLI_ROOT / "src" / "Sts2Headless" / "bin" / "Debug" / "net9.0" / "Sts2Headless.dll"
HEADLESS_RUNTIME_CONFIG = HEADLESS_DLL.with_name("Sts2Headless.runtimeconfig.json")


def headless_command():
    if HEADLESS_DLL.is_file() and HEADLESS_RUNTIME_CONFIG.is_file():
        return [DOTNET, str(HEADLESS_DLL)]
    return [DOTNET, "run", "--project", PROJECT]


def run_headless_jsonl(commands, timeout=90):
    env = os.environ.copy()
    env["DOTNET_ROOT"] = str(LOCAL_DOTNET_DIR)
    env["PATH"] = str(LOCAL_DOTNET_DIR) + os.pathsep + env.get("PATH", "")
    env["STS2_LIB"] = str(STS2_CLI_ROOT / "lib")
    env["STS2_GAME_DIR"] = str(STS2_CLI_ROOT / "lib")
    payload = "".join(json.dumps(command) + "\n" for command in commands)
    result = subprocess.run(
        headless_command(),
        input=payload,
        cwd=STS2_CLI_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    outputs = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    return result, outputs


class Game:
    """Wraps the headless C# process for testing."""

    def __init__(self):
        env = os.environ.copy()
        local_lib = STS2_CLI_ROOT / "lib"
        if LOCAL_DOTNET.is_file():
            env["DOTNET_ROOT"] = str(LOCAL_DOTNET_DIR)
            env["PATH"] = str(LOCAL_DOTNET_DIR) + os.pathsep + env.get("PATH", "")
        if (local_lib / "sts2.dll").is_file():
            env.setdefault("STS2_LIB", str(local_lib))
            env.setdefault("STS2_GAME_DIR", str(local_lib))
        else:
            env.setdefault("STS2_GAME_DIR",
                           os.path.expanduser("~/Library/Application Support/Steam/steamapps/common/"
                                              "Slay the Spire 2/SlayTheSpire2.app/Contents/Resources/"
                                              "data_sts2_macos_arm64"))
        command = headless_command()
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, env=env,
        )
        try:
            ready = self._read()
            assert ready.get("type") == "ready", f"Expected ready, got: {ready}"
        except BaseException:
            self._dispose_process(request_quit=False)
            raise

    def _read(self):
        while True:
            line = self.proc.stdout.readline().strip()
            if not line:
                raise RuntimeError("EOF from game process")
            if line.startswith("{"):
                return json.loads(line)

    def send(self, cmd):
        self.proc.stdin.write(json.dumps(cmd) + "\n")
        self.proc.stdin.flush()
        return self._read()

    def start(self, character="Ironclad", seed=DEFAULT_TEST_SEED, ascension=0, lang="en"):
        return self.send({"cmd": "start_run", "character": character,
                          "seed": seed, "ascension": ascension, "lang": lang})

    def reset(self):
        try:
            result = self.send({"cmd": "reset"})
            if result != {"type": "reset_result", "success": True}:
                raise RuntimeError(f"Reset failed: {result}")
        except Exception:
            self.close()
            self.__init__()

    def act(self, action, **args):
        cmd = {"cmd": "action", "action": action}
        if args:
            cmd["args"] = args
        return self.send(cmd)

    def get_map(self):
        return self.send({"cmd": "get_map"})

    def set_player(self, **kwargs):
        cmd = {"cmd": "set_player", **kwargs}
        return self.send(cmd)

    def enter_room(self, room_type, **kwargs):
        cmd = {"cmd": "enter_room", "type": room_type, **kwargs}
        return self.send(cmd)

    def set_draw_order(self, cards):
        return self.send({"cmd": "set_draw_order", "cards": cards})

    def _dispose_process(self, *, request_quit):
        proc = self.proc
        if request_quit and proc.poll() is None:
            try:
                proc.stdin.write('{"cmd":"quit"}\n')
                proc.stdin.flush()
            except Exception:
                pass
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait(timeout=5)
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass

    def close(self):
        self._dispose_process(request_quit=True)

    # --- Auto-play helpers ---

    @staticmethod
    def card_energy_cost(card, default=99):
        cost = card.get("energy_cost", card.get("cost", default))
        if isinstance(cost, (int, float)):
            return cost
        if isinstance(cost, str) and cost.upper() == "X":
            x_value = card.get("x_value", card.get("x_cost", 0))
            if isinstance(x_value, (int, float)):
                return x_value
            return 0
        return default

    def auto_combat(self, state):
        """Play one card or end turn."""
        hand = state.get("hand", [])
        energy = state.get("energy", 0)
        playable = [c for c in hand if c.get("can_play") and self.card_energy_cost(c) <= energy]
        if playable:
            card = playable[0]
            args = {"card_index": card["index"]}
            if card.get("target_type") == "AnyEnemy":
                enemies = state.get("enemies", [])
                if enemies:
                    args["target_index"] = enemies[0]["index"]
            return self.act("play_card", **args)
        return self.act("end_turn")

    def auto_play_combat(self, state, max_steps=300):
        """Auto-play combat until it ends."""
        for _ in range(max_steps):
            if state.get("decision") != "combat_play":
                return state
            state = self.auto_combat(state)
        raise RuntimeError("Combat did not end")

    def claim_combat_rewards(self, state, max_steps=20):
        """Claim all explicit non-card combat rewards."""
        for _ in range(max_steps):
            if state.get("decision") != "combat_reward":
                return state
            rewards = state.get("rewards", [])
            if not rewards:
                return state
            state = self.act("claim_reward", reward_index=rewards[0]["index"])
        raise RuntimeError("Combat rewards did not resolve")

    def skip_neow(self, state):
        """Skip the Neow event and all follow-up rewards until map_select."""
        for _ in range(20):
            dec = state.get("decision", "")
            if dec == "map_select":
                return state
            if dec == "event_choice":
                opts = [o for o in state["options"] if not o.get("is_locked")]
                state = self.act("choose_option", option_index=opts[0]["index"])
            elif dec == "combat_reward":
                rewards = state.get("rewards", [])
                non_card = next((r for r in rewards if r.get("kind") != "card_reward"), None)
                if non_card:
                    state = self.act("claim_reward", reward_index=non_card["index"])
                else:
                    card_reward = next((r for r in rewards if r.get("kind") == "card_reward"), None)
                    if card_reward:
                        state = self.act("skip_reward", reward_index=card_reward["index"])
                    else:
                        state = self.act("proceed")
            elif dec == "card_reward":
                state = self.act("skip_card_reward")
            elif dec == "bundle_select":
                state = self.act("select_bundle", bundle_index=0)
            elif dec == "card_select":
                if state.get("can_skip", state.get("min_select", 0) == 0):
                    state = self.act("skip_select")
                else:
                    state = self.act("select_cards", indices="0")
            else:
                state = self.act("proceed")
        return state


@pytest.fixture(scope="module")
def shared_game():
    """Reuse one headless process per module without blocking exclusive-process tests."""
    g = Game()
    yield g
    g.close()


@pytest.fixture
def game(shared_game):
    """Each test gets an isolated run state without restarting the process."""
    shared_game.reset()
    yield shared_game
    shared_game.reset()
