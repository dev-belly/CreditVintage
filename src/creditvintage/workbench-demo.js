const transitions = {
  claim: { from: "queued", status: "in_review" },
  release: { from: "in_review", status: "queued" },
  resolve: { from: "in_review", status: "resolved" },
  reopen: { from: "resolved", status: "queued" },
};

function apply(row, command) {
  if (!row || row.version !== command.expected_version) {
    const error = new Error("The case changed. Review its current state and try again.");
    error.status = 409;
    throw error;
  }
  const transition = transitions[command.action];
  if (!transition || row.status !== transition.from) throw new Error("This action is unavailable.");
  if (row.status === "in_review" && row.owner !== command.actor) {
    throw new Error("Use the current reviewer's name to finish or release this case.");
  }
  if (!command.actor || (command.action !== "claim" && !command.note)) {
    throw new Error("Enter the reviewer and any required reason.");
  }
  if (command.action === "resolve" && (!command.evidence || !["cleared", "escalated"].includes(command.outcome))) {
    throw new Error("Choose a decision and enter an evidence reference.");
  }
  row.status = transition.status;
  row.owner = transition.status === "queued" ? null : command.actor;
  row.outcome = command.action === "resolve" ? command.outcome : null;
  row.version += 1;
}

export function createDemo(onStorageLost) {
  let seed;
  let key;
  let history = [];
  let canStore = true;

  function save(events) {
    history = events;
    if (!canStore) return;
    try {
      sessionStorage.setItem(key, JSON.stringify(events));
    } catch {
      canStore = false;
      onStorageLost();
    }
  }

  function replay() {
    const rows = new Map(seed.cases.map((row) => [row.application_id, { ...row }]));
    for (const event of history) apply(rows.get(event.application_id), event);
    return { cases: Array.from(rows.values()), history, example: seed.example };
  }

  async function load() {
    if (seed) return replay();
    const response = await fetch(new URL("./queue.json", import.meta.url));
    if (!response.ok) throw new Error("The sample queue could not be loaded.");
    const data = await response.json();
    if (data.source?.data_kind !== "synthetic" || !Array.isArray(data.cases) || !data.source.queue_sha256) {
      throw new Error("The sample queue could not be verified.");
    }
    seed = data;
    key = "creditvintage-review-v1:" + seed.source.queue_sha256;

    let stored;
    try {
      stored = sessionStorage.getItem(key);
    } catch {
      canStore = false;
      onStorageLost();
    }
    if (stored) {
      try {
        history = JSON.parse(stored);
        if (!Array.isArray(history)) throw new Error("Invalid demo history");
        return replay();
      } catch {
        save([]);
      }
    }
    return replay();
  }

  async function change(command) {
    const { cases } = await load();
    const previous = history.find((event) => event.request_id === command.request_id);
    if (previous) {
      if (Object.entries(command).some(([name, value]) => previous[name] !== value)) {
        throw new Error("This request ID already belongs to another action.");
      }
      return;
    }
    apply(cases.find(({ application_id }) => application_id === command.application_id), command);
    save([...history, { ...command, occurred_at: new Date().toISOString() }]);
  }

  function reset() {
    save([]);
  }

  return { load, change, reset };
}
