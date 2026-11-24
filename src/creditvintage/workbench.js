import { createDemo } from "./workbench-demo.js";

const demoMode = document.documentElement.dataset.mode === "demo";
const byId = (id) => document.getElementById(id);
const form = byId("review");
const cases = byId("cases");
const actor = byId("actor");
const note = byId("note");
const reference = byId("reference");
const action = byId("action");
const money = new Intl.NumberFormat("en-US");
const formatPD = new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 2 });
const dateTime = new Intl.DateTimeFormat("en-US", { dateStyle: "medium", timeStyle: "short", timeZone: "UTC" });

const states = {
  queued: "Queued",
  in_review: "In review",
  resolved: "Resolved",
};
const actions = {
  queued: ["claim"],
  in_review: ["resolve", "release"],
  resolved: ["reopen"],
};
const actionLabels = {
  claim: "Claim case",
  resolve: "Resolve case",
  release: "Release case",
  reopen: "Reopen case",
};
const eventLabels = {
  claim: "Claimed",
  resolve: "Resolved",
  release: "Released",
  reopen: "Reopened",
};

let workspace;
let selected;
let busy = false;
let stateValid = false;
let pending;

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function status(text, error = false, target = "message") {
  const message = byId(target);
  message.textContent = text;
  message.dataset.error = String(error);
  message.hidden = !text;
}

async function api(path, body) {
  const options = body === undefined ? { cache: "no-store" } : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const response = await fetch(new URL("./api/" + path, location.href), options);
  const value = await response.json();
  if (!response.ok) {
    const error = new Error(value.error ?? "The workspace could not be reached.");
    error.status = response.status;
    throw error;
  }
  return value;
}

const workspaceApi = demoMode
  ? createDemo(() => {
    byId("intro-copy").textContent = "Review a sample case. Changes last while this page is open.";
  })
  : { load: () => api("state"), change: (command) => api("change", command) };

if (demoMode) {
  byId("mode").textContent = "Interactive demo · Synthetic cases";
  byId("intro-copy").textContent = "Claim a case and record a decision. Changes stay in this browser tab.";
  byId("reset").hidden = false;
  actor.value = "analyst";
}

function amountInDollars(value) {
  const cents = BigInt(value);
  return "$" + money.format(cents / 100n) + "." + String(cents % 100n).padStart(2, "0");
}

function segmentName(value) {
  return value.replaceAll("_", " ");
}

function renderSummary() {
  const counts = { queued: 0, in_review: 0, resolved: 0 };
  for (const row of workspace.cases) counts[row.status] += 1;
  const total = workspace.cases.length;
  for (const name of Object.keys(counts)) byId("count-" + name).textContent = counts[name];
  byId("completion").max = total || 1;
  byId("completion").value = counts.resolved;
  byId("progress-text").textContent = counts.resolved + " of " + total + " cases resolved";
}

function renderQueue() {
  const search = byId("search").value.trim().toLowerCase();
  const filter = byId("filter").value;
  const matching = workspace.cases.filter((row) =>
    (!filter || row.status === filter) &&
    (!search || row.application_id.toLowerCase().includes(search) || segmentName(row.segment).includes(search))
  );
  const current = matching.find((row) => row.application_id === selected?.application_id) ?? matching[0];
  updateSelected(current);

  const list = document.createDocumentFragment();
  for (const row of matching) {
    const item = node("li");
    const button = node("button", undefined, "case-button");
    const tag = node("span", states[row.status], "pill");
    tag.dataset.status = row.status;
    button.type = "button";
    button.setAttribute("aria-pressed", String(row === selected));
    button.setAttribute("aria-controls", "case-panel");
    button.append(
      node("strong", row.application_id), tag, node("small", "#" + row.review_rank + " · " + segmentName(row.segment))
    );
    button.addEventListener("click", () => {
      if (busy) return;
      cases.querySelector('[aria-pressed="true"]')?.setAttribute("aria-pressed", "false");
      button.setAttribute("aria-pressed", "true");
      updateSelected(row);
    });
    item.append(button);
    list.append(item);
  }
  cases.replaceChildren(list);
  byId("empty-queue").hidden = matching.length > 0;
  byId("match-count").textContent = matching.length + " of " + workspace.cases.length;
}

function updateSelected(row) {
  const sameCase = selected?.application_id === row?.application_id;
  selected = row;
  if (!sameCase) {
    note.value = "";
    reference.value = "";
    byId("outcome").value = "cleared";
    status("", false, "operation-message");
  }

  byId("case-title").textContent = row?.application_id ?? "Select a case";
  byId("case-state").textContent = row
    ? [row.decision_at, segmentName(row.segment), "Version " + row.version, row.owner && "Reviewer " + row.owner, row.outcome]
        .filter(Boolean).join(" · ")
    : "";
  const tag = byId("case-status");
  tag.hidden = !row;
  tag.textContent = states[row?.status] ?? "";
  tag.dataset.status = row?.status ?? "";
  byId("jump").hidden = !row;

  const details = row ? [
    ["Queue rank", "#" + row.review_rank],
    ["Calibrated PD", formatPD.format(Number(row.calibrated_pd))],
    ["Principal", amountInDollars(row.principal_cents)],
    ["Vintage", row.vintage],
  ] : [];
  const fields = byId("case-facts");
  fields.hidden = !row;
  fields.replaceChildren(...details.map(([label, value]) => {
    const item = node("div");
    item.append(node("dt", label), node("dd", value));
    return item;
  }));

  const previous = sameCase ? action.value : "";
  const available = actions[row?.status] ?? [];
  action.replaceChildren(...available.map((value) => new Option(actionLabels[value], value)));
  if (available.includes(previous)) action.value = previous;
  form.hidden = !row;
  const events = row ? workspace.history.filter((event) => event.application_id === row.application_id) : [];
  renderHistory(events);
  renderAction();
}

function renderHistory(events) {
  byId("event-count").textContent = events.length ? events.length + (events.length === 1 ? " action" : " actions") : "";
  byId("empty-history").hidden = events.length > 0;
  byId("empty-history").textContent = selected ? "No recorded actions." : "Select a case to see its history.";
  const list = document.createDocumentFragment();
  for (const event of events.toReversed()) {
    const item = node("li");
    const label = eventLabels[event.action] + " by " + event.actor;
    const time = node("time", dateTime.format(new Date(event.occurred_at)) + " UTC");
    time.dateTime = event.occurred_at;
    const timestamp = node("p");
    timestamp.append(time);
    item.append(node("strong", label), timestamp);
    if (event.outcome) item.append(node("p", event.outcome === "escalated" ? "Escalated" : "Cleared"));
    if (event.note) item.append(node("p", event.note));
    if (event.evidence) item.append(node("small", "Evidence · " + event.evidence));
    list.append(item);
  }
  byId("history").replaceChildren(list);
}

function renderAction() {
  const needsReason = action.value && action.value !== "claim";
  const resolving = action.value === "resolve";
  byId("reason-field").hidden = !needsReason;
  byId("evidence-field").hidden = !resolving;
  byId("outcome-field").hidden = !resolving;
  note.required = Boolean(needsReason);
  reference.required = resolving;
  byId("review-fields").disabled = busy || !stateValid || !selected;
  byId("submit").disabled = busy || !action.value;
}

function setBusy(value) {
  busy = value;
  for (const id of ["search", "filter"]) byId(id).disabled = busy || !workspace;
  byId("reload").disabled = busy;
  byId("reset").disabled = busy || !workspace;
  cases.setAttribute("aria-busy", String(busy));
  renderAction();
}

async function refresh({ revealSelected = false } = {}) {
  stateValid = false;
  workspace = await workspaceApi.load();
  stateValid = true;
  const current = workspace.cases.find(({ application_id }) => application_id === selected?.application_id);
  const filter = byId("filter");
  if (revealSelected && filter.value && current?.status !== filter.value) filter.value = "";
  renderSummary();
  renderQueue();
}

async function reload(firstLoad = false) {
  if (busy) return;
  setBusy(true);
  try {
    await refresh();
    status(firstLoad ? "" : "Workspace refreshed.");
  } catch (error) {
    status(error.message, true);
  } finally {
    setBusy(false);
  }
}

byId("search").addEventListener("input", renderQueue);
byId("filter").addEventListener("change", renderQueue);
action.addEventListener("change", renderAction);
byId("reload").addEventListener("click", () => void reload());
byId("reset").addEventListener("click", async () => {
  if (busy || !window.confirm("Clear this tab's sample review history?")) return;
  setBusy(true);
  try {
    workspaceApi.reset();
    byId("search").value = "";
    byId("filter").value = "";
    updateSelected(undefined);
    await refresh();
    status("Demo reset.");
  } catch (error) {
    status(error.message, true);
  } finally {
    setBusy(false);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy || !stateValid || !selected || !action.value) return;
  const command = {
    application_id: selected.application_id,
    action: action.value,
    actor: actor.value.trim(),
    expected_version: selected.version,
    note: action.value === "claim" ? "" : note.value.trim(),
    evidence: action.value === "resolve" ? reference.value.trim() : "",
    outcome: action.value === "resolve" ? byId("outcome").value : "",
  };
  if (!command.actor || (command.action !== "claim" && !command.note) || (command.action === "resolve" && !command.evidence)) {
    status("Fill in the reviewer and all required details.", true, "operation-message");
    return;
  }

  const key = JSON.stringify(command);
  if (pending?.key !== key) pending = { key, request_id: crypto.randomUUID() };
  let saved = false;
  setBusy(true);
  try {
    await workspaceApi.change({ ...command, request_id: pending.request_id });
    saved = true;
    pending = undefined;
    note.value = "";
    reference.value = "";
    await refresh({ revealSelected: true });
    status("Review saved.", false, "operation-message");
  } catch (error) {
    if (saved) {
      status("Review saved. Refresh the workspace to see the updated case.", true, "operation-message");
    } else if (error.status === 409) {
      try {
        await refresh({ revealSelected: true });
        status("The case changed. Review the updated case and try again.", true, "operation-message");
      } catch {
        status("The case changed. Refresh the workspace to see the current version.", true, "operation-message");
      }
    } else {
      status(error.message, true, "operation-message");
    }
  } finally {
    setBusy(false);
  }
});

void reload(true);
