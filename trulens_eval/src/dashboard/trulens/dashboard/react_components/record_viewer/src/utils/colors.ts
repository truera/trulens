/**
 * Types.
 *
 * Defines different typescript types related to Color.
 */
export type Color = string;
type ColorDictionary = { [k: Color]: Color };
//  Data Vizualization Color Scale
export enum DVColor {
  PRIMARY = 0,
  LIGHT = 1,
  DARK = 2,
  SECONDARY_LIGHT = 3,
  SECONDARY_DARK = 4,
}
export type DVColorScale = [Color, Color, Color, Color, Color];

/**
 * Colors.
 *
 * Defines the absolute source of the different colors used in our codebase.
 *
 * @link https://www.figma.com/file/ZdKEFr5HqxjGg2KUU51akl/%F0%9F%8C%88-Colors?node-id=958%3A47
 */

// Primary Colors
export const PRIMARY_COLOR_DARKEST: Color = '#061B22';
export const PRIMARY_COLOR_DARK: Color = '#0A2C37';
export const PRIMARY_COLOR: Color = '#2D736D';
export const PRIMARY_COLOR_LIGHT: Color = '#D3E5E4';
export const PRIMARY_COLOR_LIGHTEST: Color = '#E9F2F1';
/**
 * Secondary Colors
 * To be deprecated as this is not synced with design,
 */
export const SECONDARY_COLOR_LIGHT: Color = '#FABEB1';
export const SECONDARY_COLOR: Color = '#E0735C';
export const SECONDARY_COLOR_DARK: Color = '#A66153';
// Info Colors
export const INFO_LIGHT: Color = PRIMARY_COLOR_LIGHT;
export const INFO: Color = PRIMARY_COLOR;
export const INFO_DARK: Color = PRIMARY_COLOR_DARK;

// Navigation Colors
export const NAV_DARK: Color = PRIMARY_COLOR_DARKEST;
export const NAV_PRIMARY: Color = PRIMARY_COLOR_DARK;
export const NAV_HOVER: Color = '#133E44';
export const NAV_SELECTED: Color = PRIMARY_COLOR;

export const TRANSPARENT = 'transparent';

// UI/Default Colors
export const SUCCESS_GREEN: Color = '#4CAF50';
export const GREEN_BG: Color = '#E4F3E5';
export const WARNING_YELLOW: Color = '#F2C94C';
export const YELLOW_BG: Color = '#FDF7E4';
export const ORANGE: Color = '#FF9800';
export const ORANGE_BG: Color = '#FFF0D9';
export const RED: Color = '#EB5757';
export const RED_BG: Color = '#FCE6E6';
export const ALERT_RED: Color = '#A22C37';
export const DARK_RED: Color = '#571610';
export const NAVAJO_WHITE: Color = '#FFDBA3';

// System Colors
export const SUCCESS: Color = SUCCESS_GREEN;
export const WARNING: Color = WARNING_YELLOW;
export const WARNING_LIGHT: Color = YELLOW_BG;

// Focus Colours
export const FOCUS_YELLOW: Color = '#F6D881';
export const FOCUS_ORANGE: Color = '#F6B66A';
export const FOCUS_SALMON: Color = '#E77956';

// Data Input Format Colors
export const DATA_VIZ_BLUE: Color = '#5690C5';
export const DATA_VIZ_BLUE_BACKGROUND: Color = '#5690C515';
export const DATA_VIZ_YELLOW: Color = '#F8D06D';
export const DATA_VIZ_YELLOW_BACKGROUND: Color = '#F8D06D15';
/**
 * Gray Colors / Grey Colors
 *
 * These Grey colors are in sync with design,
 * please use these instead of importing from mui.
 */
// UI/Grey/White 00
export const WHITE: Color = '#FFFFFF';
// UI/Grey/Hover-50
export const HOVER_GRAY: Color = '#FAFAFA';
// UI/Grey/Background 100
export const BACKGROUND_GRAY: Color = '#F5F5F5';
// UI/Grey/Grey-300
export const GRAY: Color = '#E0E0E0';
// UI/Grey/Disabled-500
export const DISABLED_TEXT_GRAY: Color = '#BDBDBD';
// UI/Grey/Secondary-600
export const DARK_GRAY: Color = '#757575';
// Black-Primary-900
export const BLACK: Color = '#212121';

/**
 * Data Visualization.
 *
 * Defines the absolute source of Data Viz colors and helper functions used in our codebase.
 *
 * @link https://www.figma.com/file/ZdKEFr5HqxjGg2KUU51akl/%F0%9F%8C%88-Colors?node-id=958%3A47
 */

// Data Vizualization Colors
export const CO01: DVColorScale = ['#54A08E', '#A4CBC1', '#366567', '#7BADA4', '#1C383E'];
export const CO02: DVColorScale = ['#F8D06D', '#F0EC89', '#AD743E', '#F4E07B', '#5C291A'];
export const CO03: DVColorScale = ['#5690C5', '#8DA6BF', '#274F69', '#6D90B1', '#0B1D26'];
export const CO04: DVColorScale = ['#E77956', '#FFDBA3', '#A8402D', '#FBAD78', '#571610'];
export const CO05: DVColorScale = ['#959CFA', '#D5D1FF', '#5F74B3', '#B2B1FF', '#314A66'];
export const CO06: DVColorScale = ['#957A89', '#D2C0C4', '#664F5E', '#B59CA6', '#352731'];
export const CO07: DVColorScale = ['#78AE79', '#C7DFC3', '#5D8955', '#9FC79D', '#436036'];
export const CO08: DVColorScale = ['#FF8DA1', '#FFC9F1', '#C15F84', '#FFA9D0', '#823966'];
export const CO09: DVColorScale = ['#74B3C0', '#99D4D2', '#537F88', '#BFE6DD', '#314B50'];
export const CO10: DVColorScale = ['#A484BD', '#CBC7E4', '#745E86', '#B5A5D1', '#45384F'];

const DVColors: DVColorScale[] = [CO01, CO02, CO03, CO04, CO05, CO06, CO07, CO08, CO09, CO10];
// List of primary Datavizualization Colors
export const COLORS: Color[] = DVColors.map((c) => c[DVColor.PRIMARY]);
// Returns an Object that maps color primary dataviz colors to the selected color hue
const getColorMap = (i: DVColor): ColorDictionary =>
  Object.fromEntries(DVColors.map((c) => [c[DVColor.PRIMARY], c[i]]));
// Example: { primary_color:light hue of primary_color } or { CO01[PRIMARY]: CO01[LIGHT] }
export const LIGHT_COLORS: ColorDictionary = getColorMap(DVColor.LIGHT);
export const DARK_COLORS: ColorDictionary = getColorMap(DVColor.DARK);
export const SECONDARY_LIGHT_COLORS: ColorDictionary = getColorMap(DVColor.SECONDARY_LIGHT);
export const SECONDARY_DARK_COLORS: ColorDictionary = getColorMap(DVColor.SECONDARY_DARK);
