import { create } from "zustand";

/// UI-local state for the experiments list: the current search query and
/// pagination cursor. Navigation (which experiment / record / view is open) now
/// lives in the URL via react-router, so it is intentionally NOT stored here.
interface UiState {
  search: string;
  page: number;
  pageSize: number;
  setSearch: (search: string) => void;
  setPage: (page: number) => void;
}

export const useUiStore = create<UiState>((set) => ({
  search: "",
  page: 0,
  pageSize: 25,
  // Changing the search resets pagination to the first page.
  setSearch: (search) => set({ search, page: 0 }),
  setPage: (page) => set({ page }),
}));
