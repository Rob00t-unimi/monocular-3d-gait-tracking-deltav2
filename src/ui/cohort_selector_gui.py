# -*- coding: utf-8 -*-
"""
Interactive GUI for Selecting 60 Benchmark Videos for 3D Gait Tracking Inference.

Features:
- Organizes candidates into 6 balanced categories (10 videos per category).
- Enforces the 1-person-per-video rule (allows max 2 duplicates since unique subjects = 58 < 60).
- Prioritizes Automatic keypoints over operator-repositioned keypoints.
- Ranks candidates by Ground Truth signal quality (Optimal > Good > Imprecise).
- Filter out all discarded videos and corrupted/null sensor signals.
- Direct video playback via default system player (os.startfile).
- Smart Autofill ('Suggerisci Ottimali') to propose the top 60 balanced videos.
- Automatically exports selection to CSV and TXT upon saving or closing the window.

Usage:
    python scripts/select_inference_videos_gui.py
"""

import sys
import os
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import csv
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = PROJECT_ROOT / "data" / "candidate_videos_cache.json"
OUTPUT_CSV = PROJECT_ROOT / "data" / "selected_inference_videos_60.csv"
OUTPUT_TXT = PROJECT_ROOT / "data" / "selected_inference_videos_60.txt"

TARGET_PER_CAT = 10
TOTAL_TARGET = 60

class InferenceVideoSelectorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("3D Gait Tracking - Selezione 60 Video per Inferenza (10 per Categoria)")
        self.root.geometry("1260x780")
        self.root.minsize(1050, 650)

        # Apply ttk style
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except Exception:
            pass

        self._configure_styles()

        # Load or generate candidate data
        self.data = self._load_data()
        self.candidates_by_cat = self.data["candidates_by_cat"]
        self.category_map = self.data["category_map"]
        self.unique_persons_dataset = set(self.data["unique_persons_list"])
        self.max_unique_persons = len(self.unique_persons_dataset)
        self.unique_valid_persons = self.max_unique_persons
        self.allowed_duplicates = max(0, TOTAL_TARGET - self.max_unique_persons)

        # Selection state: map video_name -> candidate dict
        # and category_selections: map cat_key -> set of video_names
        self.selected_videos = {}
        self.category_selections = {cat_key: set() for cat_key in self.category_map}

        # Build UI layout
        self._build_header()
        self._build_notebook()
        self._build_footer()

        # Update initial counters
        self._update_status_display()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _configure_styles(self):
        self.style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        self.style.configure("Treeview", font=("Segoe UI", 9), rowheight=24)
        self.style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
        self.style.configure("Badge.TLabel", font=("Segoe UI", 9, "bold"))
        self.style.configure("Action.TButton", font=("Segoe UI", 9, "bold"))

    def _load_data(self) -> dict:
        if not CACHE_PATH.exists():
            from src.pipeline.dataset_pipeline import build_cache
            return build_cache(CACHE_PATH)
        with open(CACHE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_header(self):
        header_frame = ttk.Frame(self.root, padding=(12, 10, 12, 6))
        header_frame.pack(fill=tk.X)

        # Title & instructions
        title_box = ttk.Frame(header_frame)
        title_box.pack(fill=tk.X)

        title_lbl = ttk.Label(
            title_box,
            text="🎯 Selezione 60 Video di Benchmark per Inferenza",
            font=("Segoe UI", 13, "bold"),
            foreground="#1e3a8a"
        )
        title_lbl.pack(side=tk.LEFT)

        rules_lbl = ttk.Label(
            title_box,
            text=f"Rules: 10 videos per category | 1 subject per video (max {self.allowed_duplicates} duplicates allowed across {self.unique_valid_persons} valid subjects)",
            font=("Segoe UI", 9, "italic"),
            foreground="#4b5563"
        )
        rules_lbl.pack(side=tk.RIGHT)

        # Counter & Action Bar
        bar_frame = ttk.Frame(header_frame, padding=(0, 6, 0, 4))
        bar_frame.pack(fill=tk.X)

        self.total_lbl = tk.Label(
            bar_frame,
            text="Selezionati: 0 / 60",
            font=("Segoe UI", 11, "bold"),
            bg="#fef3c7",
            fg="#92400e",
            padx=10,
            pady=4,
            relief=tk.RIDGE
        )
        self.total_lbl.pack(side=tk.LEFT, padx=(0, 10))

        self.person_lbl = tk.Label(
            bar_frame,
            text=f"Persone Uniche: 0 / {self.max_unique_persons} (Duplicati: 0 / {self.allowed_duplicates})",
            font=("Segoe UI", 9, "bold"),
            bg="#e0f2fe",
            fg="#0369a1",
            padx=8,
            pady=4,
            relief=tk.GROOVE
        )
        self.person_lbl.pack(side=tk.LEFT, padx=5)

        # Quick action buttons on the right
        btn_box = ttk.Frame(bar_frame)
        btn_box.pack(side=tk.RIGHT)

        autofill_btn = ttk.Button(
            btn_box,
            text="✨ Suggerisci Ottimali (Autofill)",
            style="Action.TButton",
            command=self._autofill_optimal_selection
        )
        autofill_btn.pack(side=tk.LEFT, padx=4)

        reset_btn = ttk.Button(
            btn_box,
            text="🔄 Reset",
            command=self._reset_selection
        )
        reset_btn.pack(side=tk.LEFT, padx=4)

        save_btn = ttk.Button(
            btn_box,
            text="💾 Salva Selezione",
            style="Action.TButton",
            command=self._manual_save
        )
        save_btn.pack(side=tk.LEFT, padx=4)

        # Sub-bar with 6 Category Badges
        cat_badges_frame = ttk.Frame(header_frame, padding=(0, 4, 0, 2))
        cat_badges_frame.pack(fill=tk.X)

        self.cat_badges = {}
        short_names = {
            '90_R2L': '90° DX->SX',
            '90_L2R': '90° SX->DX',
            '45_R2L': '45° DX->SX',
            '45_L2R': '45° SX->DX',
            'Front': 'Frontale',
            'Rear': 'Posteriore'
        }

        for cat_key in self.category_map:
            lbl = tk.Label(
                cat_badges_frame,
                text=f"{short_names[cat_key]}: 0/{TARGET_PER_CAT}",
                font=("Segoe UI", 8, "bold"),
                bg="#f3f4f6",
                fg="#374151",
                padx=6,
                pady=2,
                relief=tk.SOLID,
                bd=1
            )
            lbl.pack(side=tk.LEFT, padx=3)
            self.cat_badges[cat_key] = lbl

    def _build_notebook(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 6))

        self.treeviews = {}
        self.search_vars = {}
        self.filter_origin_vars = {}
        self.filter_gt_vars = {}
        self.filter_cond_vars = {}
        self.filter_avail_vars = {}
        self.tab_frames = {}

        for cat_key, cat_name in self.category_map.items():
            tab_frame = ttk.Frame(self.notebook, padding=6)
            self.notebook.add(tab_frame, text=f"{cat_name} (0/{TARGET_PER_CAT})")
            self.tab_frames[cat_key] = tab_frame

            self._build_tab_content(tab_frame, cat_key)

    def _get_person_to_category_map(self) -> dict:
        """
        Returns a dict mapping person -> list of short category names where their video(s) are selected.
        """
        short_names = {
            '90_R2L': '90° DX',
            '90_L2R': '90° SX',
            '45_R2L': '45° DX',
            '45_L2R': '45° SX',
            'Front': 'Front',
            'Rear': 'Rear'
        }
        person_cats = {}
        for cat_key, vid_set in self.category_selections.items():
            for v_name in vid_set:
                c = self.selected_videos.get(v_name)
                if c:
                    p = c["person"]
                    if p not in person_cats:
                        person_cats[p] = []
                    s_name = short_names.get(cat_key, cat_key)
                    if s_name not in person_cats[p]:
                        person_cats[p].append(s_name)
        return person_cats

    def _sort_column(self, cat_key: str, col: str, reverse: bool):
        tree = self.treeviews[cat_key]
        items = [(tree.set(k, col), k) for k in tree.get_children("")]

        sel_order = {'[ ✓': 0, '[ ○': 1, '[ ⚠️': 2, '[ ⛔': 3}

        def sort_key(item):
            val = item[0]
            if col == "sel":
                return (sel_order.get(val[:3], 99), val)
            cleaned = val.replace("°", "").replace("Hz", "").strip()
            if "(score " in cleaned:
                try:
                    return float(cleaned.split("(score ")[1].replace(")", "").strip())
                except Exception:
                    pass
            try:
                return float(cleaned)
            except ValueError:
                return val.lower()

        items.sort(key=sort_key, reverse=reverse)
        for index, (_, k) in enumerate(items):
            tree.move(k, "", index)

        tree.heading(col, command=lambda: self._sort_column(cat_key, col, not reverse))

    def _build_tab_content(self, parent: ttk.Frame, cat_key: str):
        # 1. Filter bar
        filter_bar = ttk.Frame(parent, padding=(0, 0, 0, 6))
        filter_bar.pack(fill=tk.X)

        ttk.Label(filter_bar, text="Cerca:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        search_var = tk.StringVar()
        self.search_vars[cat_key] = search_var
        search_entry = ttk.Entry(filter_bar, textvariable=search_var, width=12)
        search_entry.pack(side=tk.LEFT, padx=(0, 8))
        search_var.trace_add("write", lambda *args, k=cat_key: self._refresh_tree(k))

        ttk.Label(filter_bar, text="Keypoints:").pack(side=tk.LEFT, padx=(0, 4))
        origin_var = tk.StringVar(value="All")
        self.filter_origin_vars[cat_key] = origin_var
        origin_combo = ttk.Combobox(filter_bar, textvariable=origin_var, values=["All", "Solo Automatic", "Repositioned Only"], state="readonly", width=16)
        origin_combo.pack(side=tk.LEFT, padx=(0, 8))
        origin_combo.bind("<<ComboboxSelected>>", lambda e, k=cat_key: self._refresh_tree(k))

        ttk.Label(filter_bar, text="GT Quality:").pack(side=tk.LEFT, padx=(0, 4))
        gt_var = tk.StringVar(value="All")
        self.filter_gt_vars[cat_key] = gt_var
        gt_combo = ttk.Combobox(filter_bar, textvariable=gt_var, values=["All", "Optimal/Good Only", "Optimal Only"], state="readonly", width=16)
        gt_combo.pack(side=tk.LEFT, padx=(0, 8))
        gt_combo.bind("<<ComboboxSelected>>", lambda e, k=cat_key: self._refresh_tree(k))

        ttk.Label(filter_bar, text="Abito:").pack(side=tk.LEFT, padx=(0, 4))
        cond_var = tk.StringVar(value="All")
        self.filter_cond_vars[cat_key] = cond_var
        cond_combo = ttk.Combobox(filter_bar, textvariable=cond_var, values=["All", "Standard Only (No ALT)"], state="readonly", width=18)
        cond_combo.pack(side=tk.LEFT, padx=(0, 10))
        cond_combo.bind("<<ComboboxSelected>>", lambda e, k=cat_key: self._refresh_tree(k))

        # Checkbox to hide blocked persons
        avail_var = tk.BooleanVar(value=False)
        self.filter_avail_vars[cat_key] = avail_var
        avail_cb = ttk.Checkbutton(
            filter_bar,
            text="Hide already selected subjects (show available only)",
            variable=avail_var,
            command=lambda k=cat_key: self._refresh_tree(k)
        )
        avail_cb.pack(side=tk.LEFT, padx=(6, 0))

        # 2. Table (Treeview)
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("sel", "video", "person", "origin", "gt_status", "rom", "freq", "ref_frame", "frames", "cond")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")

        tree.heading("sel", text="Selection Status", command=lambda c="sel": self._sort_column(cat_key, c, False))
        tree.heading("video", text="Video Benchmark", command=lambda c="video": self._sort_column(cat_key, c, False))
        tree.heading("person", text="Persona", command=lambda c="person": self._sort_column(cat_key, c, False))
        tree.heading("origin", text="Origine Keypoints", command=lambda c="origin": self._sort_column(cat_key, c, False))
        tree.heading("gt_status", text="GT Quality", command=lambda c="gt_status": self._sort_column(cat_key, c, False))
        tree.heading("rom", text="ROM Pitch (°)", command=lambda c="rom": self._sort_column(cat_key, c, False))
        tree.heading("freq", text="Step Cadence", command=lambda c="freq": self._sort_column(cat_key, c, False))
        tree.heading("ref_frame", text="Ref Frame", command=lambda c="ref_frame": self._sort_column(cat_key, c, False))
        tree.heading("frames", text="Durata", command=lambda c="frames": self._sort_column(cat_key, c, False))
        tree.heading("cond", text="Abbigliamento", command=lambda c="cond": self._sort_column(cat_key, c, False))

        tree.column("sel", width=165, minwidth=140, anchor=tk.CENTER)
        tree.column("video", width=210, minwidth=180, anchor=tk.W)
        tree.column("person", width=65, minwidth=55, anchor=tk.CENTER)
        tree.column("origin", width=140, minwidth=120, anchor=tk.CENTER)
        tree.column("gt_status", width=115, minwidth=100, anchor=tk.CENTER)
        tree.column("rom", width=95, minwidth=80, anchor=tk.CENTER)
        tree.column("freq", width=95, minwidth=80, anchor=tk.CENTER)
        tree.column("ref_frame", width=75, minwidth=65, anchor=tk.CENTER)
        tree.column("frames", width=65, minwidth=55, anchor=tk.CENTER)
        tree.column("cond", width=120, minwidth=100, anchor=tk.CENTER)

        # Scrollbars
        vsb = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        # Color tags: Distinct styling for selected, available, blocked, and duplicate candidates
        tree.tag_configure("row_selected", background="#dcfce7", foreground="#166534")
        tree.tag_configure("row_blocked", background="#f8fafc", foreground="#94a3b8")
        tree.tag_configure("row_dup_allowed", background="#fef9c3", foreground="#854d0e")
        tree.tag_configure("row_avail_auto", background="#ffffff", foreground="#15803d")
        tree.tag_configure("row_avail_repo", background="#ffffff", foreground="#b45309")

        # Events: double click or Space to toggle selection, Enter/P to play video
        tree.bind("<Double-1>", lambda e, k=cat_key: self._on_row_double_click(k))
        tree.bind("<space>", lambda e, k=cat_key: self._toggle_selection_selected_item(k))
        tree.bind("<Return>", lambda e, k=cat_key: self._play_selected_video(k))

        self.treeviews[cat_key] = tree

        # 3. Bottom Action toolbar for this tab
        action_bar = ttk.Frame(parent, padding=(0, 6, 0, 0))
        action_bar.pack(fill=tk.X)

        toggle_btn = ttk.Button(
            action_bar,
            text="✔ Seleziona / Deseleziona (Spazio)",
            command=lambda k=cat_key: self._toggle_selection_selected_item(k)
        )
        toggle_btn.pack(side=tk.LEFT, padx=(0, 6))

        play_btn = ttk.Button(
            action_bar,
            text="▶ Guarda Video con Player (Doppio Clic)",
            command=lambda k=cat_key: self._play_selected_video(k)
        )
        play_btn.pack(side=tk.LEFT, padx=6)

        # Legend badges
        legend_frame = ttk.Frame(action_bar)
        legend_frame.pack(side=tk.RIGHT, padx=4)

        ttk.Label(legend_frame, text="Legenda:", font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        lbl_sel = tk.Label(legend_frame, text="[✓] Selezionato", bg="#dcfce7", fg="#166534", font=("Segoe UI", 8, "bold"), padx=5, pady=1, relief=tk.SOLID, bd=1)
        lbl_sel.pack(side=tk.LEFT, padx=2)
        lbl_avail = tk.Label(legend_frame, text="[○] Disponibile", bg="#ffffff", fg="#15803d", font=("Segoe UI", 8, "bold"), padx=5, pady=1, relief=tk.SOLID, bd=1)
        lbl_avail.pack(side=tk.LEFT, padx=2)
        lbl_block = tk.Label(legend_frame, text="[⛔] In uso altrove (Bloccato)", bg="#f1f5f9", fg="#94a3b8", font=("Segoe UI", 8, "bold"), padx=5, pady=1, relief=tk.SOLID, bd=1)
        lbl_block.pack(side=tk.LEFT, padx=2)

        # Initial population
        self._refresh_tree(cat_key)

    def _build_footer(self):
        footer_frame = ttk.Frame(self.root, padding=(12, 4, 12, 8))
        footer_frame.pack(fill=tk.X)

        self.footer_status = ttk.Label(
            footer_frame,
            text="Pronto. Seleziona 10 video per categoria.",
            font=("Segoe UI", 9)
        )
        self.footer_status.pack(side=tk.LEFT)

    def _refresh_all_trees(self):
        for cat_key in self.category_map:
            self._refresh_tree(cat_key)

    def _refresh_tree(self, cat_key: str):
        tree = self.treeviews[cat_key]
        current_focus = tree.focus()

        for item in tree.get_children():
            tree.delete(item)

        candidates = self.candidates_by_cat.get(cat_key, [])
        query = self.search_vars[cat_key].get().lower().strip()
        f_origin = self.filter_origin_vars[cat_key].get()
        f_gt = self.filter_gt_vars[cat_key].get()
        f_cond = self.filter_cond_vars[cat_key].get()
        f_only_avail = self.filter_avail_vars[cat_key].get()

        person_cats = self._get_person_to_category_map()
        selected_persons = [c["person"] for c in self.selected_videos.values()]
        duplicates_used = len(selected_persons) - len(set(selected_persons))
        can_use_duplicates = duplicates_used < self.allowed_duplicates

        for cand in candidates:
            v_name = cand["video_name"]
            person = cand["person"]

            # Filter search
            if query and (query not in v_name.lower() and query not in person.lower()):
                continue

            # Filter origin
            if f_origin == "Solo Automatic" and cand["kps_origin"] not in ("Automatic", "Automatico"):
                continue
            if f_origin == "Repositioned Only" and cand["kps_origin"] in ("Automatic", "Automatico"):
                continue

            # Filter GT
            if f_gt == "Optimal/Good Only" and cand["gt_status"] not in ["Optimal", "Good"]:
                continue
            if f_gt == "Optimal Only" and cand["gt_status"] != "Optimal":
                continue

            # Filter Condition
            if f_cond == "Standard Only (No ALT)" and cand["is_alt"]:
                continue

            is_sel = v_name in self.selected_videos
            person_count = selected_persons.count(person)

            if is_sel:
                status_symbol = "[ ✓ SELEZIONATO ]"
                tag = "row_selected"
            elif person in person_cats:
                where_str = ", ".join(person_cats[person])
                if person_count >= 2 or (person_count >= 1 and not can_use_duplicates):
                    # Blocked: person already chosen elsewhere
                    if f_only_avail:
                        continue
                    status_symbol = f"[ ⛔ In uso ({where_str}) ]"
                    tag = "row_blocked"
                else:
                    # Allowed as duplicate
                    status_symbol = f"[ ⚠️ In uso ({where_str}) - Dup ]"
                    tag = "row_dup_allowed"
            else:
                status_symbol = "[ ○ Disponibile ]"
                tag = "row_avail_auto" if cand["kps_origin"] in ("Automatic", "Automatico") else "row_avail_repo"

            values = (
                status_symbol,
                v_name,
                person,
                cand["kps_origin"],
                f"{cand['gt_status']} (score {cand['gt_score']})",
                f"{cand['gt_rom']}°",
                f"{cand['gt_freq']} Hz",
                cand["ref_frame_idx"],
                cand["frames_num"],
                cand["condition"]
            )

            tree.insert("", tk.END, iid=v_name, values=values, tags=(tag,))

        if current_focus and tree.exists(current_focus):
            tree.selection_set(current_focus)
            tree.focus(current_focus)
            tree.see(current_focus)

    def _on_row_double_click(self, cat_key: str):
        tree = self.treeviews[cat_key]
        sel = tree.selection()
        if not sel:
            return
        v_name = sel[0]
        # Open video for preview
        self._play_video_by_name(v_name)

    def _play_selected_video(self, cat_key: str):
        tree = self.treeviews[cat_key]
        sel = tree.selection()
        if not sel:
            messagebox.showinfo("No Video Selected", "Select a table row before clicking.")
            return
        self._play_video_by_name(sel[0])

    def _play_video_by_name(self, video_name: str):
        cand = self._get_candidate_by_name(video_name)
        if not cand:
            return
        v_path = Path(cand["video_path"])
        if not v_path.is_absolute():
            v_path = PROJECT_ROOT / v_path
        if not v_path.exists():
            messagebox.showerror("File non trovato", f"Il file video non esiste su disco:\n{v_path}")
            return
        try:
            os.startfile(str(v_path))
            self.footer_status.config(text=f"Aperto video: {v_path.name}")
        except Exception as e:
            messagebox.showerror("Playback Error", f"Unable to launch system video player:\n{e}")

    def _get_candidate_by_name(self, video_name: str):
        for cands in self.candidates_by_cat.values():
            for c in cands:
                if c["video_name"] == video_name:
                    return c
        return None

    def _toggle_selection_selected_item(self, cat_key: str):
        tree = self.treeviews[cat_key]
        sel = tree.selection()
        if not sel:
            return
        v_name = sel[0]
        self._toggle_video_selection(cat_key, v_name)

    def _toggle_video_selection(self, cat_key: str, v_name: str):
        cand = self._get_candidate_by_name(v_name)
        if not cand:
            return

        person = cand["person"]

        # Case 1: Already selected -> Deselect
        if v_name in self.selected_videos:
            del self.selected_videos[v_name]
            self.category_selections[cat_key].discard(v_name)
            self._on_selection_changed(cat_key, v_name)
            return

        # Case 2: New selection -> Check constraints

        # Check Category limit (max 10)
        current_cat_count = len(self.category_selections[cat_key])
        if current_cat_count >= TARGET_PER_CAT:
            messagebox.showwarning(
                "Limite Categoria Raggiunto",
                    f"Category '{self.category_map[cat_key]}' already has {TARGET_PER_CAT} selected videos.\n\nDeselect a video in this category before adding a new one."
            )
            return

        # Check Total limit (max 60)
        if len(self.selected_videos) >= TOTAL_TARGET:
            messagebox.showwarning(
                "Quota Totale Raggiunta",
                    f"All {TOTAL_TARGET} target videos are already selected!\nDeselect a video before adding a new one."
            )
            return

        # Check Person constraint
        selected_persons = [c["person"] for c in self.selected_videos.values()]
        duplicates_used = len(selected_persons) - len(set(selected_persons))

        if person in selected_persons:
            existing_vids = [c["video_name"] for c in self.selected_videos.values() if c["person"] == person]
            existing_str = ", ".join(existing_vids)

            if len(existing_vids) >= 2:
                messagebox.showerror(
                    "Subject Already Selected Twice",
                        f"Cannot select '{v_name}' because subject {person} has already been selected twice in:\n{existing_str}\n\nMaximum limit: 2 videos per subject."
                )
                return

            if duplicates_used >= self.allowed_duplicates:
                messagebox.showerror(
                    "Subject Already Selected (Duplicate Limit Reached)",
                        f"Cannot select '{v_name}' because subject {person} has already been selected in:\n{existing_str}\n\nDuplicate limit ({self.allowed_duplicates}) reached."
                )
                return
            else:
                # Ask confirmation for using a duplicate slot
                confirm = messagebox.askyesno(
                    "Conferma Riutilizzo Persona",
                    f"Subject {person} is already selected in:\n{existing_str}\n\nUse one of the {self.allowed_duplicates - current_dups} remaining duplicate slots?"
                )
                if not confirm:
                    return

        # Accept selection
        self.selected_videos[v_name] = cand
        self.category_selections[cat_key].add(v_name)
        self._on_selection_changed(cat_key, v_name)

    def _on_selection_changed(self, active_cat_key: str, v_name: str):
        # Refresh all trees so blocked/available status updates dynamically across all tabs
        self._refresh_all_trees()
        # Update tab title and counters
        self._update_status_display()

    def _update_status_display(self):
        total_sel = len(self.selected_videos)
        persons = [c["person"] for c in self.selected_videos.values()]
        unique_p = len(set(persons))
        dups = total_sel - unique_p

        # Total label
        self.total_lbl.config(
            text=f"Selezionati: {total_sel} / {TOTAL_TARGET}",
            bg="#dcfce7" if total_sel == TOTAL_TARGET else ("#fef3c7" if total_sel < TOTAL_TARGET else "#fee2e2"),
            fg="#166534" if total_sel == TOTAL_TARGET else ("#92400e" if total_sel < TOTAL_TARGET else "#991b1b")
        )

        # Person label
        self.person_lbl.config(
            text=f"Persone Uniche: {unique_p} / {self.max_unique_persons} (Duplicati: {dups} / {self.allowed_duplicates})"
        )

        # Category badges and tab titles
        short_names = {
            '90_R2L': '90° DX->SX',
            '90_L2R': '90° SX->DX',
            '45_R2L': '45° DX->SX',
            '45_L2R': '45° SX->DX',
            'Front': 'Frontale',
            'Rear': 'Posteriore'
        }

        for idx, (cat_key, cat_name) in enumerate(self.category_map.items()):
            cnt = len(self.category_selections[cat_key])
            is_full = (cnt == TARGET_PER_CAT)

            # Tab title
            self.notebook.tab(idx, text=f"{cat_name} ({cnt}/{TARGET_PER_CAT})")

            # Badge
            badge = self.cat_badges[cat_key]
            badge.config(
                text=f"{short_names[cat_key]}: {cnt}/{TARGET_PER_CAT}",
                bg="#dcfce7" if is_full else "#f3f4f6",
                fg="#166534" if is_full else "#374151"
            )

        self.footer_status.config(
            text=f"Progresso: {total_sel} di {TOTAL_TARGET} video selezionati. " +
                 (f"Completato! Clicca su 'Salva Selezione'." if total_sel == TOTAL_TARGET else f"Mancano {TOTAL_TARGET - total_sel} video.")
        )

    def _autofill_optimal_selection(self):
        """
        Greedy optimization algorithm that picks the 10 best videos per category:
        1. Prioritizes Automatic keypoints.
        2. Prioritizes Optimal/Good GT quality.
        3. Prioritizes Standard clothing.
        4. Guarantees 58 unique persons + exactly 2 duplicates to reach 60 videos!
        """
        confirm = messagebox.askyesno(
            "Suggerisci Selezione Ottimale",
            "This action will automatically compute the optimal balanced selection of 60 videos:\n"
            "- 10 video per categoria\n"
            "- Priority given to automatic keypoints and highest GT quality\n"
            "- 58 persone uniche + 2 duplicati controllati\n\n"
            "The current selection will be overwritten. Do you want to proceed?"
        )
        if not confirm:
            return

        self._reset_selection(prompt=False)

        assigned_videos = {}
        person_usage = {}  # person -> count
        cat_selections = {cat: [] for cat in self.category_map}

        # Strategy:
        # First round: assign candidates to categories trying to use 1 video per person
        # Sort order for candidates inside each cat is already optimal (auto first, high GT score, etc.)
        for cat_key, cands in self.candidates_by_cat.items():
            for c in cands:
                if len(cat_selections[cat_key]) >= TARGET_PER_CAT:
                    break
                p = c["person"]
                if person_usage.get(p, 0) == 0:
                    cat_selections[cat_key].append(c)
                    person_usage[p] = 1
                    assigned_videos[c["video_name"]] = c

        # Second round: fill remaining slots up to 10 using available persons
        # Allow up to 2 duplicates total
        for cat_key, cands in self.candidates_by_cat.items():
            if len(cat_selections[cat_key]) >= TARGET_PER_CAT:
                continue
            for c in cands:
                if len(cat_selections[cat_key]) >= TARGET_PER_CAT:
                    break
                v_name = c["video_name"]
                if v_name in assigned_videos:
                    continue
                p = c["person"]
                cur_dups = sum(v - 1 for v in person_usage.values() if v > 1)
                if person_usage.get(p, 0) == 1 and cur_dups < self.allowed_duplicates:
                    cat_selections[cat_key].append(c)
                    person_usage[p] = person_usage.get(p, 0) + 1
                    assigned_videos[v_name] = c

        # Apply to GUI state
        self.selected_videos = assigned_videos
        for cat_key in self.category_map:
            self.category_selections[cat_key] = {c["video_name"] for c in cat_selections[cat_key]}

        self._refresh_all_trees()
        self._update_status_display()
        total_assigned = len(assigned_videos)
        messagebox.showinfo(
            "Selezione Ottimale Applicata",
            f"Assegnati con successo {total_assigned} video!\n"
            f"Soggetti unici utilizzati: {len(person_usage)} su {self.max_unique_persons}\n\n"
            "Puoi ora esaminare la lista in ogni tab, guardare i video desiderati e fare eventuali modifiche."
        )

    def _reset_selection(self, prompt: bool = True):
        if prompt:
            confirm = messagebox.askyesno("Confirm Reset", "Are you sure you want to deselect all videos?")
            if not confirm:
                return

        self.selected_videos.clear()
        for cat_key in self.category_map:
            self.category_selections[cat_key].clear()

        self._refresh_all_trees()
        self._update_status_display()

    def _manual_save(self):
        saved = self._save_to_files()
        if saved:
            total = len(self.selected_videos)
            messagebox.showinfo(
                "Selezione Salvata",
                f"Salvati con successo {total} video!\n\n"
                f"File CSV: {OUTPUT_CSV}\n"
                f"File TXT: {OUTPUT_TXT}"
            )

    def _save_to_files(self) -> bool:
        if not self.selected_videos:
            return False

        # 1. Export CSV
        fieldnames = [
            "video_name", "person", "experiment", "test", "sensor_position", "category_name", "angle", "direction",
            "kps_origin", "gt_status", "gt_score", "gt_rom_deg", "gt_step_freq_hz",
            "ref_frame_idx", "frames_num", "condition", "video_path", "sensor_csv"
        ]

        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            for cat_key, cat_name in self.category_map.items():
                for v_name in sorted(self.category_selections[cat_key]):
                    c = self.selected_videos[v_name]
                    writer.writerow([
                        c["video_name"], c["person"], c.get("experiment", ""), c.get("test", ""), c.get("sensor_position", "leg"),
                        c["category_name"], c["angle"], c["direction"],
                        c["kps_origin"], c["gt_status"], c["gt_score"], c["gt_rom"], c["gt_freq"],
                        c["ref_frame_idx"], c["frames_num"], c["condition"], c["video_path"], c["sensor_csv"]
                    ])

        # 2. Export TXT
        with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"  3D GAIT TRACKING: 60 BENCHMARK INFERENCE VIDEOS SELECTION\n")
            f.write(f"  Data Generazione: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Sensore IMU: Gamba Destra (T1, T2, T1ALT, T2ALT)\n")
            f.write(f"  Totale Video Selezionati: {len(self.selected_videos)} / {TOTAL_TARGET}\n")
            persons = [c["person"] for c in self.selected_videos.values()]
            f.write(f"  Persone Uniche: {len(set(persons))} (Duplicati: {len(persons) - len(set(persons))})\n")
            f.write("=" * 80 + "\n\n")

            for cat_key, cat_name in self.category_map.items():
                vids = sorted(list(self.category_selections[cat_key]))
                f.write(f"### {cat_name} ({len(vids)} / {TARGET_PER_CAT})\n")
                f.write("-" * 80 + "\n")
                for i, v_name in enumerate(vids, 1):
                    c = self.selected_videos[v_name]
                    f.write(f"{i:2d}. {c['video_name']:<22} | Persona: {c['person']:<4} | Test: {c.get('test', ''):<5} (Sensore: {c.get('sensor_position', 'leg').upper()}) | Keypoints: {c['kps_origin']:<22} | GT: {c['gt_status']:<8} (ROM {c['gt_rom']}°, {c['gt_freq']}Hz) | {c['condition']}\n")
                f.write("\n")

        print(f"Selections saved to {OUTPUT_CSV} and {OUTPUT_TXT}")
        return True

    def _on_closing(self):
        total_sel = len(self.selected_videos)
        if total_sel > 0:
            if total_sel < TOTAL_TARGET:
                confirm = messagebox.askyesnocancel(
                    "Conferma Uscita",
                    f"Hai selezionato {total_sel} di {TOTAL_TARGET} video.\n\n"
                    "Vuoi salvare la selezione parziale ed uscire?\n"
                "- Click 'Yes' to save and exit.\n"
                    "- Clicca 'No' per uscire senza salvare.\n"
                    "- Clicca 'Annulla' per continuare a selezionare."
                )
                if confirm is None:
                    return
                elif confirm:
                    self._save_to_files()
            else:
                self._save_to_files()
                messagebox.showinfo(
                    "Selezione Completata e Salvata",
                    f"All 60 videos have been saved to:\n{OUTPUT_CSV}\n{OUTPUT_TXT}"
                )

        self.root.destroy()

    on_closing = _on_closing


def main():
    root = tk.Tk()
    app = InferenceVideoSelectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()


VideoSelectorApp = InferenceVideoSelectorGUI
