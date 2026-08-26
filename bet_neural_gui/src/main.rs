// bet_neural_gui/src/main.rs — Bet Neural GUI v2
//
// Tabs
//   [⚡ Predict]     Manual match prediction + full analytics report
//   [📺 Live Board]  Auto-refreshing upcoming fixtures with predictions
//   [💰 Analytics]   Value bets overview for a gameweek
//   [🔄 Data]        Scrape / Train controls with progress log
//   [📈 Portfolio]   Bankroll & bet history
//
// Build: cd bet_neural_gui && cargo build --release

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui::{self, Color32, RichText, Ui};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ─────────────────────────────────────────────────────────────────────────────
// League
// ─────────────────────────────────────────────────────────────────────────────

#[derive(PartialEq, Clone, Copy, Debug, Default)]
enum League {
    #[default]
    PremierLeague,
    LaLiga,
    Bundesliga,
    SerieA,
    Ligue1,
    Eredivisie,
    PrimeiraLiga,
}

impl League {
    fn display_name(self) -> &'static str {
        match self {
            Self::PremierLeague         => "Premier League",
            Self::LaLiga                => "La Liga",
            Self::Bundesliga            => "Bundesliga",
            Self::SerieA                => "Serie A",
            Self::Ligue1                => "Ligue 1",
            Self::Eredivisie            => "Eredivisie",
            Self::PrimeiraLiga          => "Primeira Liga",
            Self::PrimeiraLiga  => "Primeira Liga",
        }
    }

    fn cli_key(self) -> &'static str {
        match self {
            Self::PremierLeague => "premier_league",
            Self::LaLiga        => "la_liga",
            Self::Bundesliga    => "bundesliga",
            Self::SerieA        => "serie_a",
            Self::Ligue1        => "ligue_1",
            Self::Eredivisie    => "eredivisie",
            Self::PrimeiraLiga  => "primeira_liga",
        }
    }

    fn all() -> &'static [League] {
        &[
            Self::PremierLeague, Self::LaLiga, Self::Bundesliga,
            Self::SerieA, Self::Ligue1, Self::Eredivisie, Self::PrimeiraLiga,
        ]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Prediction result (parsed from v2 CLI output)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Default, Clone)]
struct PredictionResult {
    raw_output:       String,
    // Outcome probabilities
    home_prob:        Option<f32>,
    draw_prob:        Option<f32>,
    away_prob:        Option<f32>,
    // xG
    home_xg:          Option<f32>,
    away_xg:          Option<f32>,
    // Elo
    home_elo:         Option<f32>,
    away_elo:         Option<f32>,
    // Confidence & engine
    confidence:       Option<f32>,
    engine:           String,
    // Betting
    best_bet_outcome: Option<String>,
    best_bet_odds:    Option<f32>,
    best_bet_edge:    Option<f32>,
    kelly_stake_pct:  Option<f32>,
    bm_overround:     Option<f32>,
    // Warnings / errors
    warnings:         Vec<String>,
    error:            Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Live Board
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct LiveRow {
    home_team:   String,
    away_team:   String,
    league:      String,
    kickoff_utc: String,
    is_mock:     bool,
    pred:        Option<PredictionResult>,
    loading:     bool,
    flash_until: Option<Instant>,
}

impl LiveRow {
    fn new(home: &str, away: &str, league: &str, kickoff: &str, mock: bool) -> Self {
        Self {
            home_team:   home.to_string(),
            away_team:   away.to_string(),
            league:      league.to_string(),
            kickoff_utc: kickoff.to_string(),
            is_mock:     mock,
            pred:        None,
            loading:     true,
            flash_until: None,
        }
    }

    fn secs_to_kickoff(&self) -> i64 {
        let ko  = parse_iso_to_unix(&self.kickoff_utc);
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
        ko - now
    }

    fn countdown_label(&self) -> String {
        let s = self.secs_to_kickoff();
        if s <= 0 { return "⚽ LIVE".to_string(); }
        let h = s / 3600;
        let m = (s % 3600) / 60;
        if h >= 24 { format!("{}d {:02}h", h / 24, h % 24) }
        else       { format!("{:02}:{:02}", h, m) }
    }

    fn kickoff_display(&self) -> String {
        if self.kickoff_utc.len() >= 16 {
            format!("{} {}", &self.kickoff_utc[..10], &self.kickoff_utc[11..16])
        } else {
            self.kickoff_utc.clone()
        }
    }
}

#[derive(Default)]
struct LiveBoardState {
    rows:        Vec<LiveRow>,
    refreshing:  bool,
    status_msg:  String,
    using_mock:  bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Analytics tab — value bets overview
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct ValueBet {
    home_team: String,
    away_team: String,
    outcome:   String,
    model_prob: f32,
    odds:       f32,
    edge:       f32,
    kelly_pct:  f32,
    ev:         f32,
}

#[derive(Default)]
struct AnalyticsState {
    value_bets:  Vec<ValueBet>,
    loading:     bool,
    status:      String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Data tab — scrape / train log
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Default)]
struct DataState {
    log:      Vec<String>,
    running:  bool,
    last_op:  String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Portfolio tab
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Default)]
struct PortfolioState {
    summary:    Vec<(String, String)>,   // key → value pairs
    loading:    bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab enum
// ─────────────────────────────────────────────────────────────────────────────

#[derive(PartialEq, Default)]
enum Tab { #[default] Predict, LiveBoard, Analytics, Data, Portfolio }

// ─────────────────────────────────────────────────────────────────────────────
// Application state
// ─────────────────────────────────────────────────────────────────────────────

struct BetNeuralApp {
    active_tab: Tab,
    cli_path:   String,
    python:     String,    // path to venv python or system python3

    // ── Predict tab ──
    home_team: String,
    away_team: String,
    league:    League,
    home_odds: String,
    draw_odds: String,
    away_odds: String,
    use_odds:  bool,
    full_report: bool,
    predict_result:  Arc<Mutex<Option<PredictionResult>>>,
    predict_loading: Arc<Mutex<bool>>,

    // ── Live Board ──
    board:              Arc<Mutex<LiveBoardState>>,
    board_league:       League,
    refresh_interval_s: u64,
    last_auto_refresh:  Instant,

    // ── Analytics ──
    analytics:          Arc<Mutex<AnalyticsState>>,
    analytics_league:   League,

    // ── Data ──
    data_state:     Arc<Mutex<DataState>>,
    data_league:    League,
    data_season:    String,
    scrape_all:     bool,
    train_all:      bool,

    // ── Portfolio ──
    portfolio:   Arc<Mutex<PortfolioState>>,
    bankroll_input: String,
}

impl Default for BetNeuralApp {
    fn default() -> Self {
        let script_dir = locate_script_dir();
        let cli_path   = format!("{}/bet_neural_cli.py", script_dir);
        let python     = locate_python(&script_dir);

        Self {
            active_tab:         Tab::default(),
            cli_path,
            python,
            home_team:          String::new(),
            away_team:          String::new(),
            league:             League::default(),
            home_odds:          String::new(),
            draw_odds:          String::new(),
            away_odds:          String::new(),
            use_odds:           false,
            full_report:        false,
            predict_result:     Arc::new(Mutex::new(None)),
            predict_loading:    Arc::new(Mutex::new(false)),
            board:              Arc::new(Mutex::new(LiveBoardState::default())),
            board_league:       League::default(),
            refresh_interval_s: 300,
            last_auto_refresh:  Instant::now() - Duration::from_secs(9999),
            analytics:          Arc::new(Mutex::new(AnalyticsState::default())),
            analytics_league:   League::default(),
            data_state:         Arc::new(Mutex::new(DataState::default())),
            data_league:        League::default(),
            data_season:        "2024-2025".to_string(),
            scrape_all:         false,
            train_all:          false,
            portfolio:          Arc::new(Mutex::new(PortfolioState::default())),
            bankroll_input:     "1000".to_string(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// eframe::App
// ─────────────────────────────────────────────────────────────────────────────

impl eframe::App for BetNeuralApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Live board auto-refresh
        if self.active_tab == Tab::LiveBoard {
            let elapsed    = self.last_auto_refresh.elapsed().as_secs();
            let refreshing = self.board.lock().unwrap().refreshing;
            if elapsed >= self.refresh_interval_s && !refreshing {
                self.trigger_board_refresh(ctx.clone());
            }
            ctx.request_repaint_after(Duration::from_secs(1));
        }

        // ── Top bar ───────────────────────────────────────────────────────
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading(RichText::new("🏆 Bet Neural v2").color(Color32::GOLD).size(18.0));
                ui.add_space(16.0);

                let tabs: &[(&str, Tab)] = &[
                    ("⚡ Predict",   Tab::Predict),
                    ("📺 Live Board",Tab::LiveBoard),
                    ("💰 Analytics", Tab::Analytics),
                    ("🔄 Data",      Tab::Data),
                    ("📈 Portfolio", Tab::Portfolio),
                ];
                for (label, tab) in tabs {
                    let active = std::mem::discriminant(&self.active_tab) == std::mem::discriminant(tab);
                    if ui.selectable_label(active, RichText::new(*label).size(13.0)).clicked() {
                        match tab {
                            Tab::LiveBoard => {
                                self.active_tab = Tab::LiveBoard;
                                if self.board.lock().unwrap().rows.is_empty() {
                                    self.trigger_board_refresh(ctx.clone());
                                }
                            }
                            Tab::Analytics => {
                                self.active_tab = Tab::Analytics;
                                if self.analytics.lock().unwrap().value_bets.is_empty() {
                                    self.trigger_analytics(ctx.clone());
                                }
                            }
                            Tab::Portfolio => {
                                self.active_tab = Tab::Portfolio;
                                self.load_portfolio(ctx.clone());
                            }
                            Tab::Predict  => self.active_tab = Tab::Predict,
                            Tab::Data     => self.active_tab = Tab::Data,
                        }
                    }
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let py = self.python.split('/').last().unwrap_or("python3");
                    ui.label(RichText::new(format!("engine: {} | {}", py, env!("CARGO_PKG_VERSION")))
                        .color(Color32::DARK_GRAY).size(11.0));
                });
            });
        });

        // ── Status bar ────────────────────────────────────────────────────
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                match self.active_tab {
                    Tab::Predict => {
                        let loading = *self.predict_loading.lock().unwrap();
                        if loading { ui.spinner(); ui.label("Running prediction…"); }
                        else { ui.label(RichText::new("● Ready").color(Color32::from_rgb(80,200,80)).small()); }
                    }
                    Tab::LiveBoard => {
                        let b = self.board.lock().unwrap();
                        if b.refreshing { ui.spinner(); ui.label("Fetching…"); }
                        else {
                            let mock = if b.using_mock { " ⚙ mock" } else { " 🌐 live" };
                            ui.label(RichText::new(format!("{}{}", b.status_msg, mock)).color(Color32::GRAY).small());
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let next = self.refresh_interval_s.saturating_sub(self.last_auto_refresh.elapsed().as_secs());
                            ui.label(RichText::new(format!("refresh in {}s", next)).color(Color32::DARK_GRAY).small());
                        });
                    }
                    Tab::Analytics => {
                        let a = self.analytics.lock().unwrap();
                        if a.loading { ui.spinner(); ui.label("Scanning for value bets…"); }
                        else { ui.label(RichText::new(format!("💰 {} value bets found", a.value_bets.len())).small()); }
                    }
                    Tab::Data => {
                        let d = self.data_state.lock().unwrap();
                        if d.running { ui.spinner(); ui.label(format!("Running {}…", d.last_op)); }
                        else { ui.label(RichText::new("● Idle").color(Color32::GRAY).small()); }
                    }
                    Tab::Portfolio => {
                        ui.label(RichText::new("📈 Portfolio").small());
                    }
                }
            });
        });

        // ── Central panel ─────────────────────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.active_tab {
                Tab::Predict   => self.draw_predict(ui, ctx),
                Tab::LiveBoard => self.draw_live_board(ui, ctx),
                Tab::Analytics => self.draw_analytics(ui, ctx),
                Tab::Data      => self.draw_data(ui, ctx),
                Tab::Portfolio => self.draw_portfolio(ui),
            }
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Predict tab
// ─────────────────────────────────────────────────────────────────────────────

impl BetNeuralApp {
    fn draw_predict(&mut self, ui: &mut Ui, ctx: &egui::Context) {
        egui::SidePanel::left("predict_input").resizable(false).min_width(220.0).show_inside(ui, |ui| {
            ui.add_space(8.0);
            ui.label(RichText::new("Match").strong().size(14.0));
            ui.separator();
            ui.add_space(4.0);

            ui.label("🏠 Home team");
            ui.text_edit_singleline(&mut self.home_team);
            ui.add_space(4.0);
            ui.label("✈️  Away team");
            ui.text_edit_singleline(&mut self.away_team);
            ui.add_space(8.0);

            ui.label(RichText::new("League").strong());
            egui::ComboBox::from_id_source("p_league")
                .selected_text(self.league.display_name())
                .show_ui(ui, |ui| {
                    for &l in League::all() {
                        ui.selectable_value(&mut self.league, l, l.display_name());
                    }
                });

            ui.add_space(8.0);
            ui.checkbox(&mut self.use_odds, "Provide odds");
            if self.use_odds {
                ui.add_space(4.0);
                ui.label("Home odds"); ui.text_edit_singleline(&mut self.home_odds);
                ui.label("Draw odds"); ui.text_edit_singleline(&mut self.draw_odds);
                ui.label("Away odds"); ui.text_edit_singleline(&mut self.away_odds);
            }

            ui.add_space(6.0);
            ui.checkbox(&mut self.full_report, "Full analytics report");

            ui.add_space(12.0);
            let loading = *self.predict_loading.lock().unwrap();
            let can_go  = !loading
                && !self.home_team.trim().is_empty()
                && !self.away_team.trim().is_empty();

            if ui.add_enabled(can_go,
                egui::Button::new(RichText::new("⚡ Predict").size(15.0))
                    .min_size(egui::vec2(190.0, 36.0))).clicked()
            {
                self.run_predict(ctx.clone());
            }

            if loading { ui.spinner(); }
        });

        // Right panel — results
        ui.vertical(|ui| {
            let guard = self.predict_result.lock().unwrap();
            match guard.as_ref() {
                None => {
                    ui.vertical_centered(|ui| {
                        ui.add_space(80.0);
                        ui.label(RichText::new("Enter teams and press ⚡ Predict")
                            .color(Color32::GRAY).size(15.0));
                        ui.add_space(16.0);
                        ui.label(RichText::new("Enable \"Full analytics report\" for value betting analysis,\nstorylines, and a complete intelligence breakdown.")
                            .color(Color32::DARK_GRAY).size(12.0));
                    });
                }
                Some(r) if r.error.is_some() => {
                    ui.colored_label(Color32::RED, "❌ Prediction error");
                    ui.separator();
                    ui.label(r.error.as_deref().unwrap_or("Unknown"));
                    ui.collapsing("Raw output", |ui| ui.monospace(&r.raw_output));
                }
                Some(r) => {
                    for w in &r.warnings {
                        ui.colored_label(Color32::from_rgb(255,200,50), w);
                    }
                    if !r.warnings.is_empty() { ui.separator(); }

                    ui.horizontal(|ui| {
                        ui.heading(format!("{} vs {}", self.home_team.trim(), self.away_team.trim()));
                        ui.add_space(10.0);
                        ui.label(RichText::new(self.league.display_name()).color(Color32::GOLD));
                    });
                    if !r.engine.is_empty() {
                        ui.label(RichText::new(format!("🔧 {}", r.engine))
                            .color(Color32::DARK_GRAY).size(11.0));
                    }
                    ui.separator();
                    ui.add_space(4.0);

                    // Probabilities
                    ui.label(RichText::new("🎲 Probabilities").strong());
                    ui.add_space(4.0);
                    for (label, prob, col) in [
                        ("🏠 Home Win", r.home_prob, Color32::from_rgb(70,130,220)),
                        ("🤝 Draw",     r.draw_prob, Color32::from_rgb(160,160,60)),
                        ("✈️  Away Win", r.away_prob, Color32::from_rgb(220,80,80)),
                    ] {
                        draw_prob_bar(ui, label, prob, col);
                        ui.add_space(2.0);
                    }

                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if let (Some(hxg), Some(axg)) = (r.home_xg, r.away_xg) {
                            ui.label(format!("⚽ xG: {:.2} – {:.2}", hxg, axg));
                        }
                        if let (Some(he), Some(ae)) = (r.home_elo, r.away_elo) {
                            ui.add_space(12.0);
                            ui.label(RichText::new(format!("📊 Elo: {:.0} – {:.0}", he, ae))
                                .color(Color32::GRAY));
                        }
                        if let Some(conf) = r.confidence {
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.colored_label(confidence_color(conf),
                                    format!("🔥 {:.1}%", conf * 100.0));
                            });
                        }
                    });

                    // Betting section
                    if r.best_bet_outcome.is_some() || r.kelly_stake_pct.is_some() {
                        ui.add_space(10.0);
                        ui.separator();
                        ui.label(RichText::new("💰 Betting").strong());
                        ui.add_space(4.0);

                        if let Some(ref bm_or) = r.bm_overround {
                            ui.label(RichText::new(format!("Bookmaker overround: {:.1}%", bm_or))
                                .color(Color32::GRAY).size(11.0));
                        }

                        if let Some(ref best) = r.best_bet_outcome {
                            ui.colored_label(Color32::from_rgb(80,220,80), format!("⭐ Best bet: {}", best));
                            if let (Some(odds), Some(edge)) = (r.best_bet_odds, r.best_bet_edge) {
                                ui.horizontal(|ui| {
                                    ui.label(format!("  Odds: {:.2}", odds));
                                    ui.add_space(8.0);
                                    ui.colored_label(
                                        if edge > 0.0 { Color32::from_rgb(80,220,80) } else { Color32::RED },
                                        format!("Edge: {:+.1}%", edge * 100.0),
                                    );
                                });
                            }
                            if let Some(k) = r.kelly_stake_pct {
                                ui.label(format!("  💸 Kelly stake: {:.2}% of bankroll", k));
                            }
                        } else {
                            ui.colored_label(Color32::GRAY, "❌ No value bets at current odds");
                        }
                    }

                    // Full analytics report
                    if !r.raw_output.is_empty() {
                        ui.add_space(10.0);
                        ui.collapsing("📋 Full report / raw output", |ui| {
                            egui::ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
                                ui.monospace(&r.raw_output);
                            });
                        });
                    }
                }
            }
        });
    }

    fn run_predict(&self, ctx: egui::Context) {
        let home   = self.home_team.trim().to_string();
        let away   = self.away_team.trim().to_string();
        let league = self.league.cli_key().to_string();
        let python = self.python.clone();
        let cli    = self.cli_path.clone();
        let odds   = if self.use_odds {
            Some(format!("{},{},{}", self.home_odds.trim(), self.draw_odds.trim(), self.away_odds.trim()))
        } else { None };
        let full_report = self.full_report;

        *self.predict_loading.lock().unwrap() = true;
        *self.predict_result.lock().unwrap()  = None;

        let result_arc  = Arc::clone(&self.predict_result);
        let loading_arc = Arc::clone(&self.predict_loading);

        thread::spawn(move || {
            let cmd_name = if full_report { "analytics" } else { "predict" };
            let mut cmd = Command::new(&python);
            cmd.arg(&cli).arg(cmd_name).arg(format!("{} vs {}", home, away))
               .arg("--league").arg(&league);
            if let Some(ref o) = odds { cmd.arg("--odds").arg(o); }

            let r = match cmd.output() {
                Err(e) => PredictionResult {
                    error: Some(format!("Launch failed: {}", e)),
                    ..Default::default()
                },
                Ok(out) => {
                    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                    if !out.status.success() {
                        PredictionResult {
                            raw_output: stdout.clone(),
                            error: Some(if stderr.is_empty() { stdout } else { stderr }),
                            ..Default::default()
                        }
                    } else {
                        parse_cli_output(&stdout)
                    }
                }
            };
            *result_arc.lock().unwrap()  = Some(r);
            *loading_arc.lock().unwrap() = false;
            ctx.request_repaint();
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Live Board tab
// ─────────────────────────────────────────────────────────────────────────────

impl BetNeuralApp {
    fn trigger_board_refresh(&mut self, ctx: egui::Context) {
        self.last_auto_refresh = Instant::now();
        let board  = Arc::clone(&self.board);
        let python = self.python.clone();
        let cli    = self.cli_path.clone();
        let league = self.board_league.cli_key().to_string();
        let league_display = self.board_league.display_name().to_string();

        { let mut b = board.lock().unwrap(); b.refreshing = true; b.status_msg = "Fetching fixtures…".into(); }

        thread::spawn(move || {
            let fixtures = fetch_fixtures(&python, &league);
            {
                let mut b = board.lock().unwrap();
                b.rows = fixtures.iter().map(|(h, a, ko, m)| {
                    LiveRow::new(h, a, &league_display, ko, *m)
                }).collect();
                b.using_mock = fixtures.iter().any(|f| f.3);
            }
            ctx.request_repaint();

            let count = fixtures.len();
            for (idx, (home, away, _, _)) in fixtures.into_iter().enumerate() {
                let raw = run_cli_sync(&python, &cli, &["predict", &format!("{} vs {}", home, away), "--league", &league]);
                let pred = parse_cli_output(&raw);
                let mut b = board.lock().unwrap();
                if let Some(row) = b.rows.get_mut(idx) {
                    row.pred    = Some(pred);
                    row.loading = false;
                    row.flash_until = Some(Instant::now() + Duration::from_millis(1500));
                }
                ctx.request_repaint();
            }

            let mut b = board.lock().unwrap();
            b.refreshing = false;
            b.status_msg = format!("{} fixtures loaded", count);
            ctx.request_repaint();
        });
    }

    fn draw_live_board(&mut self, ui: &mut Ui, ctx: &egui::Context) {
        // Toolbar
        ui.horizontal(|ui| {
            ui.label(RichText::new("League:").strong());
            let prev = self.board_league;
            egui::ComboBox::from_id_source("b_league")
                .selected_text(self.board_league.display_name())
                .show_ui(ui, |ui| {
                    for &l in League::all() { ui.selectable_value(&mut self.board_league, l, l.display_name()); }
                });
            if prev != self.board_league { self.trigger_board_refresh(ctx.clone()); }

            ui.add_space(10.0);
            ui.label("Auto refresh:");
            egui::ComboBox::from_id_source("b_interval")
                .selected_text(format!("{}m", self.refresh_interval_s / 60))
                .show_ui(ui, |ui| {
                    for (s, l) in [(60,"1m"),(300,"5m"),(600,"10m"),(1800,"30m")] {
                        ui.selectable_value(&mut self.refresh_interval_s, s, l);
                    }
                });

            ui.add_space(10.0);
            let refreshing = self.board.lock().unwrap().refreshing;
            if ui.add_enabled(!refreshing, egui::Button::new("🔄 Now")).clicked() {
                self.trigger_board_refresh(ctx.clone());
            }
            if refreshing { ui.spinner(); }
        });

        ui.add_space(4.0);
        ui.separator();

        // Header
        egui::Grid::new("bh").num_columns(9).min_col_width(55.0).striped(false).show(ui, |ui| {
            for h in ["Kick-off","Match","H Win%","Draw%","A Win%","Conf","xG","Engine","Best Bet"] {
                ui.label(RichText::new(h).strong().color(Color32::GOLD).size(11.0));
            }
            ui.end_row();
        });
        ui.separator();

        let now = Instant::now();
        egui::ScrollArea::vertical().show(ui, |ui| {
            let mut board = self.board.lock().unwrap();
            if board.rows.is_empty() && !board.refreshing {
                ui.vertical_centered(|ui| {
                    ui.add_space(40.0);
                    ui.label(RichText::new("No fixtures. Press 🔄 Now to load.").color(Color32::GRAY));
                });
                return;
            }

            egui::Grid::new("br").num_columns(9).min_col_width(55.0).min_row_height(24.0).striped(true).show(ui, |ui| {
                for row in &board.rows {
                    let flash = row.flash_until.map(|t| t > now).unwrap_or(false);
                    let bg = if flash { Color32::from_rgba_unmultiplied(60,200,60,25) } else { Color32::TRANSPARENT };

                    // Kick-off
                    let secs = row.secs_to_kickoff();
                    let t_col = if secs <= 0 { Color32::from_rgb(255,100,100) }
                               else if secs < 3600 { Color32::YELLOW }
                               else { Color32::GRAY };
                    ui.vertical(|ui| {
                        ui.label(RichText::new(row.kickoff_display()).size(10.0).color(Color32::GRAY));
                        ui.label(RichText::new(row.countdown_label()).size(10.0).color(t_col));
                    });

                    // Match
                    let mock_flag = if row.is_mock { " ⚙" } else { "" };
                    ui.label(RichText::new(format!("{} – {}{}", row.home_team, row.away_team, mock_flag))
                        .size(12.0).background_color(bg));

                    if row.loading {
                        for _ in 0..7 { ui.label("…"); }
                    } else {
                        let p = row.pred.as_ref();
                        draw_mini_bar(ui, p.and_then(|r| r.home_prob), Color32::from_rgb(70,130,220));
                        draw_mini_bar(ui, p.and_then(|r| r.draw_prob), Color32::from_rgb(160,160,60));
                        draw_mini_bar(ui, p.and_then(|r| r.away_prob), Color32::from_rgb(220,80,80));

                        if let Some(c) = p.and_then(|r| r.confidence) {
                            ui.label(RichText::new(format!("{:.0}%", c * 100.0))
                                .size(11.0).color(confidence_color(c)).strong());
                        } else { ui.label("—"); }

                        if let (Some(hxg), Some(axg)) = (
                            p.and_then(|r| r.home_xg), p.and_then(|r| r.away_xg)
                        ) {
                            ui.label(RichText::new(format!("{:.1}–{:.1}", hxg, axg)).size(10.0).color(Color32::GRAY));
                        } else { ui.label("—"); }

                        if let Some(eng) = p.map(|r| &r.engine) {
                            let short = if eng.contains("ml") { "ML+Elo" } else { "Elo" };
                            ui.label(RichText::new(short).size(10.0).color(Color32::DARK_GRAY));
                        } else { ui.label("—"); }

                        if let Some(best) = p.and_then(|r| r.best_bet_outcome.as_ref()) {
                            let edge_str = p.and_then(|r| r.best_bet_edge)
                                .map(|e| format!(" ({:+.0}%)", e * 100.0))
                                .unwrap_or_default();
                            ui.label(RichText::new(format!("{}{}", best, edge_str))
                                .size(10.0).color(Color32::from_rgb(80,220,80)));
                        } else {
                            ui.label(RichText::new("—").size(10.0).color(Color32::DARK_GRAY));
                        }
                    }
                    ui.end_row();
                }
            });
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Analytics tab
// ─────────────────────────────────────────────────────────────────────────────

impl BetNeuralApp {
    fn trigger_analytics(&self, ctx: egui::Context) {
        let python  = self.python.clone();
        let cli     = self.cli_path.clone();
        let league  = self.analytics_league.cli_key().to_string();
        let arc     = Arc::clone(&self.analytics);

        { let mut a = arc.lock().unwrap(); a.loading = true; a.status = "Scanning gameweek…".into(); }

        thread::spawn(move || {
            // Run gameweek command which prints all matches
            let raw = run_cli_sync(&python, &cli, &["gameweek", "--league", &league]);
            let value_bets = parse_value_bets_from_gameweek(&raw);
            let mut a = arc.lock().unwrap();
            a.value_bets = value_bets;
            a.loading    = false;
            a.status     = format!("{} value bets detected", a.value_bets.len());
            ctx.request_repaint();
        });
    }

    fn draw_analytics(&mut self, ui: &mut Ui, ctx: &egui::Context) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Gameweek Value Scanner").strong().size(15.0));
            ui.add_space(12.0);
            ui.label("League:");
            let prev = self.analytics_league;
            egui::ComboBox::from_id_source("a_league")
                .selected_text(self.analytics_league.display_name())
                .show_ui(ui, |ui| {
                    for &l in League::all() { ui.selectable_value(&mut self.analytics_league, l, l.display_name()); }
                });
            if prev != self.analytics_league { self.trigger_analytics(ctx.clone()); }

            ui.add_space(10.0);
            let loading = self.analytics.lock().unwrap().loading;
            if ui.add_enabled(!loading, egui::Button::new("🔍 Scan")).clicked() {
                self.trigger_analytics(ctx.clone());
            }
            if loading { ui.spinner(); }
        });

        ui.add_space(4.0);
        ui.separator();

        let a = self.analytics.lock().unwrap();

        if a.loading {
            ui.vertical_centered(|ui| { ui.add_space(60.0); ui.spinner(); ui.label("Scanning for value bets…"); });
            return;
        }

        if a.value_bets.is_empty() {
            ui.vertical_centered(|ui| {
                ui.add_space(60.0);
                ui.label(RichText::new("No value bets found in current gameweek.").color(Color32::GRAY).size(13.0));
                ui.label(RichText::new("Tip: scrape & train models for sharper edges.").color(Color32::DARK_GRAY).size(11.0));
            });
            return;
        }

        ui.label(RichText::new(format!("⭐ {} Value Bets Found — {}", a.value_bets.len(), self.analytics_league.display_name()))
            .color(Color32::GOLD).size(14.0));
        ui.add_space(6.0);

        egui::Grid::new("vb").num_columns(7).min_col_width(70.0).striped(true).show(ui, |ui| {
            for h in ["Match","Outcome","Model%","Odds","Edge","Kelly%","EV"] {
                ui.label(RichText::new(h).strong().color(Color32::GOLD).size(11.0));
            }
            ui.end_row();

            for vb in &a.value_bets {
                ui.label(RichText::new(format!("{} vs {}", vb.home_team, vb.away_team)).size(11.0));
                ui.label(RichText::new(&vb.outcome).size(11.0).color(Color32::from_rgb(80,200,80)));
                ui.label(format!("{:.1}%", vb.model_prob * 100.0));
                ui.label(format!("{:.2}", vb.odds));
                let edge_col = if vb.edge > 0.05 { Color32::from_rgb(80,220,80) }
                               else if vb.edge > 0.0 { Color32::YELLOW }
                               else { Color32::RED };
                ui.colored_label(edge_col, format!("{:+.1}%", vb.edge * 100.0));
                ui.label(format!("{:.2}%", vb.kelly_pct));
                ui.label(RichText::new(format!("{:+.3}", vb.ev))
                    .color(if vb.ev > 0.0 { Color32::from_rgb(80,220,80) } else { Color32::RED }));
                ui.end_row();
            }
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Data tab
// ─────────────────────────────────────────────────────────────────────────────

impl BetNeuralApp {
    fn draw_data(&mut self, ui: &mut Ui, ctx: &egui::Context) {
        ui.label(RichText::new("Data Pipeline").strong().size(15.0));
        ui.add_space(4.0);
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("League:");
            egui::ComboBox::from_id_source("d_league")
                .selected_text(self.data_league.display_name())
                .show_ui(ui, |ui| {
                    for &l in League::all() { ui.selectable_value(&mut self.data_league, l, l.display_name()); }
                });
            ui.add_space(8.0);
            ui.label("Season:");
            ui.text_edit_singleline(&mut self.data_season);
        });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.scrape_all, "Scrape all leagues");
            ui.checkbox(&mut self.train_all,  "Train all leagues");
        });

        ui.add_space(8.0);
        let running = self.data_state.lock().unwrap().running;

        ui.horizontal(|ui| {
            // Scrape button
            if ui.add_enabled(!running, egui::Button::new("🌐 Scrape Data")
                .min_size(egui::vec2(130.0, 32.0))).clicked()
            {
                self.run_data_op("scrape", ctx.clone());
            }

            ui.add_space(8.0);

            // Train button
            if ui.add_enabled(!running, egui::Button::new("🧠 Train Models")
                .min_size(egui::vec2(130.0, 32.0))).clicked()
            {
                self.run_data_op("train", ctx.clone());
            }

            ui.add_space(8.0);

            // Status
            if ui.add_enabled(!running, egui::Button::new("📂 Status")).clicked() {
                self.run_data_op("status", ctx.clone());
            }

            if running { ui.spinner(); }
        });

        ui.add_space(8.0);
        ui.separator();
        ui.label(RichText::new("Log").strong());
        ui.add_space(4.0);

        egui::ScrollArea::vertical().max_height(340.0).stick_to_bottom(true).show(ui, |ui| {
            let ds = self.data_state.lock().unwrap();
            for line in &ds.log {
                let col = if line.contains("✅") { Color32::from_rgb(80,200,80) }
                          else if line.contains("❌") { Color32::RED }
                          else if line.contains("⚠") { Color32::YELLOW }
                          else { Color32::GRAY };
                ui.label(RichText::new(line).size(11.0).color(col).monospace());
            }
        });
    }

    fn run_data_op(&self, op: &str, ctx: egui::Context) {
        let python  = self.python.clone();
        let cli     = self.cli_path.clone();
        let league  = self.data_league.cli_key().to_string();
        let season  = self.data_season.clone();
        let all_l   = if op == "scrape" { self.scrape_all } else { self.train_all };
        let op_str  = op.to_string();
        let arc     = Arc::clone(&self.data_state);

        {
            let mut d = arc.lock().unwrap();
            d.running  = true;
            d.last_op  = op_str.clone();
            d.log.push(format!("▶ {} {} {} …", op_str, league, season));
        }

        thread::spawn(move || {
            let mut args = vec![op_str.as_str()];
            let l_arg; let s_arg;
            if op_str == "status" {
                // status doesn't take any arguments
            } else if all_l {
                l_arg = "--all".to_string();
                args.push("--all");
            } else {
                l_arg = format!("{}", league);
                args.push("--league");
                args.push(&l_arg);
            }
            if op_str == "scrape" || op_str == "train" {
                s_arg = format!("{}", season);
                args.push("--season");
                args.push(&s_arg);
            } else {
                s_arg = String::new();
                let _ = s_arg;
            }

            let output = Command::new(&python)
                .arg(&cli)
                .args(&args)
                .output();

            let mut d = arc.lock().unwrap();
            match output {
                Err(e) => {
                    d.log.push(format!("❌ Launch error: {}", e));
                }
                Ok(out) => {
                    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                    for line in stdout.lines() {
                        if !line.trim().is_empty() {
                            d.log.push(line.to_string());
                        }
                    }
                    if !out.status.success() {
                        let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                        for line in stderr.lines() {
                            if !line.trim().is_empty() {
                                d.log.push(format!("⚠ {}", line));
                            }
                        }
                    }
                }
            }
            d.log.push("─".repeat(50));
            d.running = false;
            ctx.request_repaint();
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Portfolio tab
// ─────────────────────────────────────────────────────────────────────────────

impl BetNeuralApp {
    fn load_portfolio(&self, ctx: egui::Context) {
        let python = self.python.clone();
        let cli    = self.cli_path.clone();
        let arc    = Arc::clone(&self.portfolio);

        { arc.lock().unwrap().loading = true; }

        thread::spawn(move || {
            let raw = run_cli_sync(&python, &cli, &["portfolio", "summary"]);
            let summary = raw.lines()
                .filter(|l| l.contains(":") || l.contains("="))
                .map(|l| {
                    let parts: Vec<&str> = l.splitn(2, ':').collect();
                    if parts.len() == 2 {
                        (parts[0].trim().to_string(), parts[1].trim().to_string())
                    } else {
                        (l.trim().to_string(), String::new())
                    }
                })
                .collect::<Vec<_>>();

            let mut p = arc.lock().unwrap();
            p.summary = summary;
            p.loading = false;
            ctx.request_repaint();
        });
    }

    fn draw_portfolio(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Portfolio").strong().size(15.0));
        });
        ui.add_space(4.0);
        ui.separator();

        let p = self.portfolio.lock().unwrap();
        if p.loading {
            ui.vertical_centered(|ui| { ui.add_space(60.0); ui.spinner(); });
            return;
        }

        if p.summary.is_empty() {
            ui.label(RichText::new("No portfolio data. Place some bets first.").color(Color32::GRAY));
        } else {
            egui::Grid::new("port").num_columns(2).min_col_width(140.0).striped(true).show(ui, |ui| {
                for (k, v) in &p.summary {
                    ui.label(RichText::new(k).strong().size(12.0));
                    let col = if v.contains('+') { Color32::from_rgb(80,200,80) }
                              else if v.contains('-') { Color32::RED }
                              else { Color32::WHITE };
                    ui.colored_label(col, v);
                    ui.end_row();
                }
            });
        }

        ui.add_space(12.0);
        ui.separator();
        ui.label(RichText::new("Bankroll Management").strong());
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("Bankroll: £");
            ui.text_edit_singleline(&mut self.bankroll_input);
        });
        ui.add_space(4.0);
        ui.label(RichText::new("Max single bet: 5% of bankroll (Kelly-capped)").color(Color32::GRAY).size(11.0));
        ui.label(RichText::new("Max total exposure: 25% of bankroll").color(Color32::GRAY).size(11.0));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI helpers
// ─────────────────────────────────────────────────────────────────────────────

fn run_cli_sync(python: &str, cli: &str, args: &[&str]) -> String {
    Command::new(python)
        .arg(cli)
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default()
}

fn fetch_fixtures(python: &str, league: &str) -> Vec<(String, String, String, bool)> {
    let script = locate_script_dir() + "/football_data_api.py";
    let out = Command::new(python)
        .arg(&script).arg("fixtures").arg("--league").arg(league).arg("--days").arg("14")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();

    out.lines().filter_map(|line| {
        let line = line.trim();
        let is_mock = line.contains('⚙');
        let clean   = line.trim_end_matches('⚙').trim();
        let parts: Vec<&str> = clean.splitn(3, ' ').collect();
        if parts.len() < 3 { return None; }
        if !parts[2].contains(" vs ") { return None; }
        let teams: Vec<&str> = parts[2].splitn(2, " vs ").collect();
        if teams.len() != 2 { return None; }
        let kickoff = format!("{}T{}:00Z", parts[0], parts[1]);
        Some((teams[0].trim().to_string(), teams[1].trim().to_string(), kickoff, is_mock))
    }).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Output parsers
// ─────────────────────────────────────────────────────────────────────────────

fn parse_cli_output(output: &str) -> PredictionResult {
    let mut r = PredictionResult { raw_output: output.to_string(), ..Default::default() };

    for line in output.lines() {
        let t = line.trim();

        // Probabilities — look for lines with emoji medals or "Home Win" / "Away Win" / "Draw"
        if t.contains("Home Win") {
            r.home_prob = parse_pct(t);
        } else if t.contains("Away Win") {
            r.away_prob = parse_pct(t);
        } else if t.contains("Draw") && !t.to_uppercase().contains("DRAW:") {
            if r.draw_prob.is_none() { r.draw_prob = parse_pct(t); }
        }

        // xG
        if t.contains("EXPECTED GOALS") || (t.starts_with('⚽') && t.contains(':')) {
            if let Some(xg_part) = t.split(':').nth(1) {
                for sep in [" – ", " - ", "–", "-"] {
                    if xg_part.contains(sep) {
                        let p: Vec<&str> = xg_part.trim().splitn(2, sep).collect();
                        if p.len() == 2 {
                            r.home_xg = p[0].trim().parse().ok();
                            r.away_xg = p[1].split_whitespace().next().and_then(|v| v.parse().ok());
                            break;
                        }
                    }
                }
            }
        }

        // Elo
        if t.contains("🏠") && t.contains(':') {
            let nums: Vec<f32> = t.split_whitespace()
                .filter_map(|w| w.trim_matches(|c: char| !c.is_ascii_digit() && c != '.').parse().ok())
                .collect();
            if let Some(&v) = nums.last() { if v > 1000.0 { r.home_elo = Some(v); } }
        }
        if t.contains("✈") && t.contains(':') {
            let nums: Vec<f32> = t.split_whitespace()
                .filter_map(|w| w.trim_matches(|c: char| !c.is_ascii_digit() && c != '.').parse().ok())
                .collect();
            if let Some(&v) = nums.last() { if v > 1000.0 { r.away_elo = Some(v); } }
        }

        // Confidence
        if t.contains("CONFIDENCE") { r.confidence = parse_pct(t); }

        // Engine
        if t.contains("Engine:") || t.contains("🔧") {
            if let Some(e) = t.split(':').nth(1) {
                r.engine = e.trim().to_string();
            }
        }

        // Best bet
        if t.contains("BEST BET") || t.starts_with("⭐") {
            // ⭐ BEST BET: Home Win @ 2.10  (edge: +8.2%, kelly: 2.50%)
            if let Some(after) = t.split(':').nth(1) {
                let parts: Vec<&str> = after.trim().splitn(2, '@').collect();
                r.best_bet_outcome = Some(parts[0].trim().to_string());
                if parts.len() > 1 {
                    r.best_bet_odds = parts[1].trim().split_whitespace().next()
                        .and_then(|v| v.trim_matches(|c: char| c == ',' || c == ')').parse().ok());
                }
            }
            r.best_bet_edge = parse_signed_pct(t);
            if let Some(k) = parse_after_keyword(t, "kelly") { r.kelly_stake_pct = Some(k); }
        }

        // Kelly (standalone)
        if (t.contains("Kelly") || t.contains("kelly")) && r.kelly_stake_pct.is_none() {
            r.kelly_stake_pct = parse_after_keyword(t, "kelly");
        }

        // Overround
        if t.contains("overround") {
            r.bm_overround = parse_pct(t).map(|v| v * 100.0);
        }

        // Warnings
        if t.contains('⚠') { r.warnings.push(t.to_string()); }

        // Errors
        if t.contains("Error") || t.contains("error") {
            r.error = Some(t.to_string());
        }
    }

    r
}

fn parse_value_bets_from_gameweek(output: &str) -> Vec<ValueBet> {
    let mut bets = Vec::new();
    let mut current_home = String::new();
    let mut current_away = String::new();

    for line in output.lines() {
        let t = line.trim();

        if (t.contains("🤖") || t.contains("📊")) && t.contains(" vs ") {
            if let Some(match_part) = t.split(|c: char| c.is_whitespace()).skip(1).next() {
                let _ = match_part;
            }
            // Parse "⚽ Arsenal vs Chelsea" or "🤖 Arsenal vs Chelsea"
            let clean = t.trim_start_matches(|c: char| !c.is_alphabetic());
            if clean.contains(" vs ") {
                let teams: Vec<&str> = clean.splitn(2, " vs ").collect();
                if teams.len() == 2 {
                    current_home = teams[0].trim().to_string();
                    current_away = teams[1].trim().to_string();
                }
            }
        }

        if t.contains("Value:") || t.contains("💰 Value:") {
            // "💰 Value: Home Win @ 2.40  edge +6.2%"
            if let Some(after) = t.split("Value:").nth(1) {
                let parts: Vec<&str> = after.trim().splitn(2, '@').collect();
                if parts.len() == 2 {
                    let outcome = parts[0].trim().to_string();
                    let rest    = parts[1];
                    let odds: f32    = rest.trim().split_whitespace().next()
                        .and_then(|v| v.parse().ok()).unwrap_or(0.0);
                    let edge: f32    = parse_signed_pct(rest).unwrap_or(0.0);
                    if odds > 1.0 {
                        bets.push(ValueBet {
                            home_team:  current_home.clone(),
                            away_team:  current_away.clone(),
                            outcome,
                            model_prob: 0.0,
                            odds,
                            edge,
                            kelly_pct:  edge.max(0.0) / (odds - 1.0) * 50.0, // half-kelly approx
                            ev:         odds * 0.0 - (1.0 - 0.0),
                        });
                    }
                }
            }
        }
    }
    bets
}

fn parse_pct(s: &str) -> Option<f32> {
    for token in s.split_whitespace() {
        let clean = token.trim_end_matches('%').trim_matches(|c: char| c == '(' || c == ')');
        if let Ok(v) = clean.parse::<f32>() {
            if v >= 0.0 && v <= 100.0 { return Some(v / 100.0); }
        }
    }
    None
}

fn parse_signed_pct(s: &str) -> Option<f32> {
    for token in s.split_whitespace() {
        let clean = token.trim_end_matches('%').trim_matches(|c: char| c == '(' || c == ')');
        if let Ok(v) = clean.parse::<f32>() {
            if v >= -100.0 && v <= 100.0 { return Some(v / 100.0); }
        }
    }
    None
}

fn parse_after_keyword(s: &str, keyword: &str) -> Option<f32> {
    let lower = s.to_lowercase();
    let idx   = lower.find(keyword)?;
    let rest  = &s[idx + keyword.len()..];
    parse_pct(rest)
}

// ─────────────────────────────────────────────────────────────────────────────
// Drawing helpers
// ─────────────────────────────────────────────────────────────────────────────

fn confidence_color(conf: f32) -> Color32 {
    if conf >= 0.65 { Color32::from_rgb(80,200,80) }
    else if conf >= 0.55 { Color32::YELLOW }
    else { Color32::from_rgb(200,80,80) }
}

fn draw_prob_bar(ui: &mut Ui, label: &str, prob: Option<f32>, color: Color32) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).size(13.0));
        if let Some(p) = prob {
            let available = ui.available_width() - 65.0;
            let bar_w = available.max(40.0);
            let filled = (p * bar_w).max(2.0);
            let (rect, _) = ui.allocate_exact_size(egui::vec2(bar_w, 16.0), egui::Sense::hover());
            let painter = ui.painter();
            painter.rect_filled(egui::Rect::from_min_size(rect.min, egui::vec2(bar_w, 16.0)), 4.0, Color32::from_gray(40));
            painter.rect_filled(egui::Rect::from_min_size(rect.min, egui::vec2(filled, 16.0)), 4.0, color);
            ui.label(format!("{:.1}%", p * 100.0));
        } else {
            ui.label("—");
        }
    });
}

fn draw_mini_bar(ui: &mut Ui, prob: Option<f32>, color: Color32) {
    if let Some(p) = prob {
        ui.vertical(|ui| {
            let bar_w = 52.0;
            let (rect, _) = ui.allocate_exact_size(egui::vec2(bar_w, 9.0), egui::Sense::hover());
            let painter = ui.painter();
            painter.rect_filled(egui::Rect::from_min_size(rect.min, egui::vec2(bar_w, 9.0)), 3.0, Color32::from_gray(35));
            painter.rect_filled(egui::Rect::from_min_size(rect.min, egui::vec2((p * bar_w).max(2.0), 9.0)), 3.0, color);
            ui.label(RichText::new(format!("{:.0}%", p * 100.0)).size(10.0));
        });
    } else {
        ui.label("—");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Path / environment helpers
// ─────────────────────────────────────────────────────────────────────────────

fn locate_script_dir() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("../../..").canonicalize().ok()).flatten())
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .to_string_lossy()
        .to_string()
}

fn locate_python(script_dir: &str) -> String {
    // Prefer venv
    for candidate in [
        format!("{}/venv/bin/python3", script_dir),
        format!("{}/venv/bin/python", script_dir),
    ] {
        if std::path::Path::new(&candidate).exists() {
            return candidate;
        }
    }
    "python3".to_string()
}

fn parse_iso_to_unix(s: &str) -> i64 {
    if s.len() < 19 { return 0; }
    let y: i64  = s[0..4].parse().unwrap_or(2026);
    let mo: i64 = s[5..7].parse().unwrap_or(1);
    let d: i64  = s[8..10].parse().unwrap_or(1);
    let h: i64  = s[11..13].parse().unwrap_or(0);
    let mi: i64 = s[14..16].parse().unwrap_or(0);
    let sc: i64 = s[17..19].parse().unwrap_or(0);
    days_since_epoch(y, mo, d) * 86400 + h * 3600 + mi * 60 + sc
}

fn days_since_epoch(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let m = month;
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Bet Neural v2")
            .with_inner_size([1100.0, 640.0])
            .with_min_inner_size([800.0, 480.0]),
        ..Default::default()
    };

    eframe::run_native("Bet Neural v2", options, Box::new(|cc| {
        let mut style = (*cc.egui_ctx.style()).clone();
        style.visuals = egui::Visuals::dark();
        // Slightly larger default font
        let mut fonts = egui::FontDefinitions::default();
        for (_, d) in fonts.font_data.iter_mut() {
            let _ = d;
        }
        cc.egui_ctx.set_style(style);
        Ok(Box::<BetNeuralApp>::default())
    }))
}
