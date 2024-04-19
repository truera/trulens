import KeyboardArrowDownRounded from '@mui/icons-material/KeyboardArrowDownRounded';
import KeyboardArrowUpRounded from '@mui/icons-material/KeyboardArrowUpRounded';
import { SimpleTreeView } from '@mui/x-tree-view';
import { Streamlit } from 'streamlit-component-lib';
import RecordTreeCellRecursive from './RecordTreeCellRecursive';
import { StackTreeNode, ROOT_NODE_ID } from '../utils/StackTreeNode';

type RecordTreeProps = {
  nodeMap: Record<string, StackTreeNode>;
  root: StackTreeNode;
  selectedNodeId: string | null;
  setSelectedNodeId: (newId: string | null) => void;
};

export default function RecordTree({ nodeMap, root, selectedNodeId, setSelectedNodeId }: RecordTreeProps) {
  const handleItemSelectionToggle = (_event: React.SyntheticEvent, itemId: string, isSelected: boolean) => {
    if (isSelected) {
      setSelectedNodeId(itemId);
    } else {
      setSelectedNodeId(null);
    }
  };

  const { timeTaken: totalTime, startTime: treeStart } = root;

  return (
    <SimpleTreeView
      sx={{
        p: 1,
        overflowY: 'auto',
        flexGrow: 0,
        [`& > li`]: {
          minWidth: 'fit-content',
        },
      }}
      slots={{
        collapseIcon: KeyboardArrowUpRounded,
        expandIcon: KeyboardArrowDownRounded,
      }}
      onExpandedItemsChange={() => {
        // Add a delay - streamlit is not great at detecting height changes due to animation, so we wait
        // until the end of the animation to tell streamlit to set the frame height.
        setTimeout(() => Streamlit.setFrameHeight(), 300);
      }}
      defaultSelectedItems={selectedNodeId ?? ROOT_NODE_ID}
      defaultExpandedItems={Object.keys(nodeMap) ?? []}
      onItemSelectionToggle={handleItemSelectionToggle}
    >
      <RecordTreeCellRecursive node={root} depth={0} totalTime={totalTime} treeStart={treeStart} />
    </SimpleTreeView>
  );
}
