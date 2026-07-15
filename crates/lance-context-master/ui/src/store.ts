import { create } from "zustand";

/// UI-local view state: the current search query and pagination cursor.
interface UiState {
  view: "experiments" | "tasks";
  search: string;
  page: number;
  pageSize: number;
  selected: string | null;
  setView: (view: "experiments" | "tasks") => void;
  setSearch: (search: string) => void;
  setPage: (page: number) => void;
  select: (name: string | null) => void;
}

export const useUiStore = create<UiState>((set) => ({
  view: "experiments",
  search: "",
  page: 0,
  pageSize: 25,
  selected: null,
  setView: (view) => set({ view }),
  // Changing the search resets pagination to the first page.
  setSearch: (search) => set({ search, page: 0 }),
  setPage: (page) => set({ page }),
  select: (selected) => set({ selected }),
}));
